//! Prospective Memory - Future intentions and reminders (SHO-116)
//!
//! Implements the "remembering to remember" capability:
//! - Time-based triggers (at specific time, after duration)
//! - Context-based triggers (keyword match, semantic similarity)
//!
//! Architecture:
//! - ProspectiveTask stored in "prospective" column family of shared RocksDB
//! - Secondary indices in "prospective_index" column family
//! - Memory with ExperienceType::Intention created for semantic integration
//! - Uses Hebbian learning for decay (same as regular memories)

use anyhow::{Context, Result};
use chrono::Utc;
use rocksdb::{ColumnFamilyDescriptor, Options, WriteBatch, DB};
use std::path::Path;
use std::sync::Arc;

use super::types::{ProspectiveTask, ProspectiveTaskId, ProspectiveTaskStatus, ProspectiveTrigger};

/// Column family for main task storage (key = `{user_id}:{task_id}`)
const CF_PROSPECTIVE: &str = "prospective";
/// Column family for secondary indices (due dates, status, keyword lookups)
const CF_PROSPECTIVE_INDEX: &str = "prospective_index";

/// Compute cosine similarity between two embedding vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Migrate unpadded `due:{ts}:{id}` keys to zero-padded `due:{:020}:{id}` format.
///
/// Prior versions wrote bare timestamps (e.g. `due:1739404800:uuid`), which break
/// lexicographic ordering (`"9" > "10"`). Zero-padding to 20 digits ensures
/// lex order = chronological order, enabling early-termination scans.
fn migrate_due_key_padding(store: &dyn crate::storage::ColumnStore) -> Result<usize> {
    // A logical key rewrite, not a rocksdb-to-rocksdb import — so it runs on the
    // trait and applies to whichever backend holds the rows.
    let mut batch = crate::storage::WriteBatch::new();
    let mut count = 0;

    for (key, value) in store.scan_cf(CF_PROSPECTIVE_INDEX, Some(b"due:"), None)? {
        let key_str = std::str::from_utf8(&key).context("Non-UTF8 key in prospective due index")?;

        // Key format: due:{timestamp}:{task_id}
        let parts: Vec<&str> = key_str.splitn(3, ':').collect();
        if parts.len() != 3 {
            continue;
        }

        // Already padded — nothing to do
        if parts[1].len() >= 20 {
            continue;
        }

        if let Ok(ts) = parts[1].parse::<i64>() {
            let new_key = format!("due:{:020}:{}", ts, parts[2]);
            batch.delete_cf(CF_PROSPECTIVE_INDEX, key);
            batch.put_cf(CF_PROSPECTIVE_INDEX, new_key.as_bytes(), value);
            count += 1;
        }
    }

    if count > 0 {
        store
            .write_batch(&batch)
            .context("Failed to write migrated prospective due keys")?;
        tracing::info!(count, "Migrated prospective due keys to zero-padded format");
    }

    Ok(count)
}

/// Storage and query engine for prospective memory (reminders)
pub struct ProspectiveStore {
    /// Two families — CF_PROSPECTIVE and CF_PROSPECTIVE_INDEX — so this takes the
    /// multi-family handle rather than a KeyValueStore bound to one. update() and
    /// delete() batch across BOTH families in one commit; the DB-scoped batch API
    /// exists for exactly those two call sites.
    store: Arc<dyn crate::storage::ColumnStore>,
    /// Present ONLY when rocksdb-backed: migrate_from_separate_dbs (a one-time
    /// rocksdb-to-rocksdb import) and the rocksdb backup accessor need it; a
    /// store built over another backend has no engine DB to hold.
    db: Option<Arc<DB>>,
}

impl ProspectiveStore {
    /// Return the column family descriptors needed by ProspectiveStore.
    ///
    /// Call this when opening the shared DB so the CFs are registered.
    pub fn column_family_descriptors() -> Vec<ColumnFamilyDescriptor> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        vec![
            ColumnFamilyDescriptor::new(CF_PROSPECTIVE, opts.clone()),
            ColumnFamilyDescriptor::new(CF_PROSPECTIVE_INDEX, opts),
        ]
    }

    /// Create a new prospective store backed by the given shared DB.
    ///
    /// The DB must have been opened with the column families returned by
    /// [`column_family_descriptors()`].  On first run after the migration,
    /// data from the old separate `tasks/` and `index/` sub-DBs is copied
    /// into the corresponding CFs and the old directories are renamed.
    pub fn new(db: Arc<DB>, storage_path: &Path) -> Result<Self> {
        let prospective_path = storage_path.join("prospective");
        std::fs::create_dir_all(&prospective_path)?;

        Self::migrate_from_separate_dbs(&prospective_path, &db)?;

        let store: Arc<dyn crate::storage::ColumnStore> = Arc::new(
            crate::storage::legacy_rocksdb::RocksDbColumnStore::new(Arc::clone(&db)),
        );
        migrate_due_key_padding(store.as_ref())?;

        tracing::info!("Prospective memory store initialized");
        Ok(Self {
            store,
            db: Some(db),
        })
    }

    /// Bind a handle directly — the constructor a backend-agnostic caller uses.
    /// Skips migrate_from_separate_dbs (a rocksdb-to-rocksdb import with no
    /// meaning on another backend) but still runs the logical due-key
    /// migration, which belongs to whichever backend holds the rows.
    pub fn with_store(store: Arc<dyn crate::storage::ColumnStore>) -> Result<Self> {
        for cf in [CF_PROSPECTIVE, CF_PROSPECTIVE_INDEX] {
            store.ensure_cf(cf)?;
        }
        migrate_due_key_padding(store.as_ref())?;
        tracing::info!("Prospective memory store initialized");
        Ok(Self { store, db: None })
    }

    /// One-time migration: copy data from legacy separate RocksDB instances
    /// (`prospective/tasks/` and `prospective/index/`) into the shared DB's
    /// column families, then rename the old directories so we don't re-migrate.
    fn migrate_from_separate_dbs(prospective_path: &Path, db: &DB) -> Result<()> {
        let old_dirs: &[(&str, &str)] =
            &[("tasks", CF_PROSPECTIVE), ("index", CF_PROSPECTIVE_INDEX)];

        for (old_name, cf_name) in old_dirs {
            let old_dir = prospective_path.join(old_name);
            if !old_dir.is_dir() {
                continue;
            }

            let cf = db
                .cf_handle(cf_name)
                .unwrap_or_else(|| panic!("{cf_name} CF must exist"));
            let old_opts = Options::default();
            match DB::open_for_read_only(&old_opts, &old_dir, false) {
                Ok(old_db) => {
                    let mut batch = WriteBatch::default();
                    let mut count = 0usize;
                    for (key, value) in old_db.iterator(rocksdb::IteratorMode::Start).flatten() {
                        batch.put_cf(cf, &key, &value);
                        count += 1;
                        if count.is_multiple_of(10_000) {
                            db.write(std::mem::take(&mut batch))?;
                        }
                    }
                    if !batch.is_empty() {
                        db.write(batch)?;
                    }
                    drop(old_db);
                    tracing::info!(
                        "  prospective/{old_name}: migrated {count} entries to {cf_name} CF"
                    );

                    let backup = prospective_path.join(format!("{old_name}.pre_cf_migration"));
                    if backup.exists() {
                        let _ = std::fs::remove_dir_all(&backup);
                    }
                    if let Err(e) = std::fs::rename(&old_dir, &backup) {
                        tracing::warn!("Could not rename old {old_name} dir: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("Could not open old {old_name} DB for migration: {e}");
                }
            }
        }
        Ok(())
    }

    /// Flush all column families to disk (critical for graceful shutdown)
    pub fn flush(&self) -> Result<()> {
        // One flush per family, errors propagated — the old loop did the same with
        // flush_cf_opt(set_wait(true)); the trait's flush is synchronous.
        for cf_name in [CF_PROSPECTIVE, CF_PROSPECTIVE_INDEX] {
            self.store
                .flush_cf(cf_name)
                .map_err(|e| anyhow::anyhow!("Failed to flush {cf_name}: {e}"))?;
        }
        Ok(())
    }

    /// Get references to all RocksDB databases for backup.
    ///
    /// Empty when not rocksdb-backed: there is no engine DB to hand a rocksdb
    /// backup engine, and pretending otherwise is how a backup omits a store
    /// while reporting success.
    pub fn databases(&self) -> Vec<(&str, &Arc<DB>)> {
        match &self.db {
            Some(db) => vec![("prospective_shared", db)],
            None => Vec::new(),
        }
    }

    /// Store a new prospective task
    pub fn store(&self, task: &ProspectiveTask) -> Result<()> {
        let key = format!("{}:{}", task.user_id, task.id);
        // Use JSON instead of bincode - handles tagged enums properly and is human-readable
        let value =
            serde_json::to_vec(task).context("Failed to serialize prospective task to JSON")?;

        self.store
            .put_cf(CF_PROSPECTIVE, key.as_bytes(), &value)
            .context("Failed to store prospective task")?;

        // Update indices
        self.update_indices(task)?;

        tracing::debug!(
            task_id = %task.id,
            user_id = %task.user_id,
            trigger = ?task.trigger,
            "Stored prospective task"
        );

        Ok(())
    }

    /// Update secondary indices for efficient queries
    fn update_indices(&self, task: &ProspectiveTask) -> Result<()> {
        let mut batch = crate::storage::WriteBatch::new();

        // Index by user (for listing user's reminders)
        let user_key = format!("user:{}:{}", task.user_id, task.id);
        batch.put_cf(CF_PROSPECTIVE_INDEX, user_key.as_bytes(), &b"1"[..]);

        // Index by status
        let status_key = format!("status:{:?}:{}:{}", task.status, task.user_id, task.id);
        batch.put_cf(CF_PROSPECTIVE_INDEX, status_key.as_bytes(), &b"1"[..]);

        // Index by due time (for time-based trigger queries)
        // Zero-padded to 20 digits for correct lexicographic ordering
        if let Some(due_at) = task.trigger.due_at() {
            let due_key = format!("due:{:020}:{}", due_at.timestamp(), task.id);
            batch.put_cf(
                CF_PROSPECTIVE_INDEX,
                due_key.as_bytes(),
                task.user_id.as_bytes(),
            );
        }

        // Index context triggers by keywords
        if let ProspectiveTrigger::OnContext { ref keywords, .. } = task.trigger {
            for keyword in keywords {
                let kw_key = format!(
                    "context:{}:{}:{}",
                    keyword.to_lowercase(),
                    task.user_id,
                    task.id
                );
                batch.put_cf(CF_PROSPECTIVE_INDEX, kw_key.as_bytes(), &b"1"[..]);
            }
        }

        self.store
            .write_batch(&batch)
            .context("Failed to update prospective indices")?;

        Ok(())
    }

    /// Get a task by ID
    pub fn get(
        &self,
        user_id: &str,
        task_id: &ProspectiveTaskId,
    ) -> Result<Option<ProspectiveTask>> {
        let key = format!("{}:{}", user_id, task_id);

        match self.store.get_cf(CF_PROSPECTIVE, key.as_bytes())? {
            Some(value) => {
                let task: ProspectiveTask = serde_json::from_slice(&value)
                    .context("Failed to deserialize prospective task from JSON")?;
                Ok(Some(task))
            }
            None => Ok(None),
        }
    }

    /// Update a task (e.g., mark as triggered/dismissed)
    ///
    /// Atomic: reads old task, builds a single WriteBatch containing old-index
    /// deletes + new task write + new index writes, then commits once.
    pub fn update(&self, task: &ProspectiveTask) -> Result<()> {
        // ONE batch across CF_PROSPECTIVE and CF_PROSPECTIVE_INDEX — the DB-scoped
        // batch API exists for this site; two batches would open a window where the
        // task and its indices disagree, invisible from the call site.
        let mut batch = crate::storage::WriteBatch::new();

        // 1. Remove old indices (need to read old task to know what to delete)
        if let Some(old_task) = self.get(&task.user_id, &task.id)? {
            let user_key = format!("user:{}:{}", task.user_id, task.id);
            batch.delete_cf(CF_PROSPECTIVE_INDEX, user_key.as_bytes());

            let status_key = format!("status:{:?}:{}:{}", old_task.status, task.user_id, task.id);
            batch.delete_cf(CF_PROSPECTIVE_INDEX, status_key.as_bytes());

            if let Some(due_at) = old_task.trigger.due_at() {
                let due_key = format!("due:{:020}:{}", due_at.timestamp(), task.id);
                batch.delete_cf(CF_PROSPECTIVE_INDEX, due_key.as_bytes());
            }

            if let ProspectiveTrigger::OnContext { ref keywords, .. } = old_task.trigger {
                for keyword in keywords {
                    let kw_key = format!(
                        "context:{}:{}:{}",
                        keyword.to_lowercase(),
                        task.user_id,
                        task.id
                    );
                    batch.delete_cf(CF_PROSPECTIVE_INDEX, kw_key.as_bytes());
                }
            }
        }

        // 2. Write new task data
        let key = format!("{}:{}", task.user_id, task.id);
        let value =
            serde_json::to_vec(task).context("Failed to serialize prospective task to JSON")?;
        batch.put_cf(CF_PROSPECTIVE, key.as_bytes(), value.as_slice());

        // 3. Write new indices
        let user_key = format!("user:{}:{}", task.user_id, task.id);
        batch.put_cf(CF_PROSPECTIVE_INDEX, user_key.as_bytes(), &b"1"[..]);

        let status_key = format!("status:{:?}:{}:{}", task.status, task.user_id, task.id);
        batch.put_cf(CF_PROSPECTIVE_INDEX, status_key.as_bytes(), &b"1"[..]);

        // Only index due time for Pending tasks — triggered/dismissed tasks
        // don't need to appear in due scans (fixes M13: stale due index bloat)
        if task.status == ProspectiveTaskStatus::Pending {
            if let Some(due_at) = task.trigger.due_at() {
                let due_key = format!("due:{:020}:{}", due_at.timestamp(), task.id);
                batch.put_cf(
                    CF_PROSPECTIVE_INDEX,
                    due_key.as_bytes(),
                    task.user_id.as_bytes(),
                );
            }
        }

        if let ProspectiveTrigger::OnContext { ref keywords, .. } = task.trigger {
            for keyword in keywords {
                let kw_key = format!(
                    "context:{}:{}:{}",
                    keyword.to_lowercase(),
                    task.user_id,
                    task.id
                );
                batch.put_cf(CF_PROSPECTIVE_INDEX, kw_key.as_bytes(), &b"1"[..]);
            }
        }

        // 4. Single atomic commit
        self.store
            .write_batch(&batch)
            .context("Failed to atomically update prospective task")?;

        tracing::debug!(
            task_id = %task.id,
            user_id = %task.user_id,
            status = ?task.status,
            "Updated prospective task (atomic)"
        );

        Ok(())
    }

    /// Delete a task (atomic: removes task + all indices in single WriteBatch)
    pub fn delete(&self, user_id: &str, task_id: &ProspectiveTaskId) -> Result<bool> {
        let task = match self.get(user_id, task_id)? {
            Some(t) => t,
            None => return Ok(false),
        };

        // ONE batch across both families — see update() for why this must not split.
        let mut batch = crate::storage::WriteBatch::new();

        // Delete task data
        let key = format!("{}:{}", user_id, task_id);
        batch.delete_cf(CF_PROSPECTIVE, key.as_bytes());

        // Delete all indices
        let user_key = format!("user:{}:{}", user_id, task_id);
        batch.delete_cf(CF_PROSPECTIVE_INDEX, user_key.as_bytes());

        let status_key = format!("status:{:?}:{}:{}", task.status, user_id, task_id);
        batch.delete_cf(CF_PROSPECTIVE_INDEX, status_key.as_bytes());

        if let Some(due_at) = task.trigger.due_at() {
            let due_key = format!("due:{:020}:{}", due_at.timestamp(), task_id);
            batch.delete_cf(CF_PROSPECTIVE_INDEX, due_key.as_bytes());
        }

        if let ProspectiveTrigger::OnContext { ref keywords, .. } = task.trigger {
            for keyword in keywords {
                let kw_key = format!("context:{}:{}:{}", keyword.to_lowercase(), user_id, task_id);
                batch.delete_cf(CF_PROSPECTIVE_INDEX, kw_key.as_bytes());
            }
        }

        self.store
            .write_batch(&batch)
            .context("Failed to atomically delete prospective task")?;

        tracing::debug!(task_id = %task_id, user_id = %user_id, "Deleted prospective task");

        Ok(true)
    }

    /// List all tasks for a user, optionally filtered by status
    pub fn list_for_user(
        &self,
        user_id: &str,
        status_filter: Option<ProspectiveTaskStatus>,
    ) -> Result<Vec<ProspectiveTask>> {
        let prefix = format!("user:{}:", user_id);
        let mut tasks = Vec::new();

        // scan_cf enforces the prefix bound.
        for (key, _) in self
            .store
            .scan_cf(CF_PROSPECTIVE_INDEX, Some(prefix.as_bytes()), None)?
        {
            let key_str = String::from_utf8_lossy(&key);

            // Extract task_id from key: user:{user_id}:{task_id}
            if let Some(task_id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(task_id_str) {
                    let task_id = ProspectiveTaskId(uuid);
                    if let Some(task) = self.get(user_id, &task_id)? {
                        // Apply status filter if specified
                        if let Some(filter) = status_filter {
                            if task.status == filter {
                                tasks.push(task);
                            }
                        } else {
                            tasks.push(task);
                        }
                    }
                }
            }
        }

        // Sort by created_at descending (newest first)
        tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(tasks)
    }

    /// Get all due time-based tasks for a user
    ///
    /// Returns tasks where:
    /// - Trigger is time-based (AtTime or AfterDuration)
    /// - Trigger time <= now
    /// - Status is Pending
    pub fn get_due_tasks(&self, user_id: &str) -> Result<Vec<ProspectiveTask>> {
        let now = Utc::now();
        let now_ts = now.timestamp();

        let mut due_tasks = Vec::new();

        // Scan due index for tasks with due_time <= now
        // Key format: due:{timestamp}:{task_id}
        // NOTE: the break below is an ORDERING early-exit, not a prefix guard — it
        // relies on scan_cf returning keys in lexicographic order (zero-padded
        // timestamps make lex order chronological). It stays.
        for (key, value) in self.store.scan_cf(CF_PROSPECTIVE_INDEX, Some(b"due:"), None)? {
            let key_str = String::from_utf8_lossy(&key);

            // Parse key: due:{timestamp}:{task_id}
            let parts: Vec<&str> = key_str.splitn(3, ':').collect();
            if parts.len() != 3 {
                continue;
            }

            let task_ts: i64 = match parts[1].parse() {
                Ok(ts) => ts,
                Err(_) => continue,
            };

            // With zero-padded keys, lexicographic order = chronological order.
            // All remaining keys are also in the future — stop scanning.
            if task_ts > now_ts {
                break;
            }

            // Check user matches
            let stored_user_id = String::from_utf8_lossy(&value);
            if stored_user_id != user_id {
                continue;
            }

            // Get task and check status
            if let Ok(uuid) = uuid::Uuid::parse_str(parts[2]) {
                let task_id = ProspectiveTaskId(uuid);
                if let Some(task) = self.get(user_id, &task_id)? {
                    if task.status == ProspectiveTaskStatus::Pending {
                        due_tasks.push(task);
                    }
                }
            }
        }

        // Sort by priority (higher first) then by due time (earliest first)
        due_tasks.sort_by(|a, b| {
            let priority_cmp = b.priority.cmp(&a.priority);
            if priority_cmp != std::cmp::Ordering::Equal {
                return priority_cmp;
            }
            let a_due = a.trigger.due_at().unwrap_or(a.created_at);
            let b_due = b.trigger.due_at().unwrap_or(b.created_at);
            a_due.cmp(&b_due)
        });

        Ok(due_tasks)
    }

    /// Scan ALL users for due reminders (used by active reminder scheduler).
    ///
    /// Returns `(user_id, task)` pairs for all pending tasks whose due time has passed.
    /// Leverages the zero-padded `due:{timestamp}:{task_id}` index for efficient scanning:
    /// lexicographic order = chronological, so we stop at the first future timestamp.
    pub fn get_all_due_tasks(&self) -> Result<Vec<(String, ProspectiveTask)>> {
        let now_ts = Utc::now().timestamp();
        let mut due_tasks = Vec::new();

        // Same ordering early-exit as get_due_tasks — the break is load-bearing.
        for (key, value) in self.store.scan_cf(CF_PROSPECTIVE_INDEX, Some(b"due:"), None)? {
            let key_str = String::from_utf8_lossy(&key);

            let parts: Vec<&str> = key_str.splitn(3, ':').collect();
            if parts.len() != 3 {
                continue;
            }

            let task_ts: i64 = match parts[1].parse() {
                Ok(ts) => ts,
                Err(_) => continue,
            };

            if task_ts > now_ts {
                break;
            }

            let user_id = String::from_utf8_lossy(&value).to_string();

            if let Ok(uuid) = uuid::Uuid::parse_str(parts[2]) {
                let task_id = ProspectiveTaskId(uuid);
                if let Some(task) = self.get(&user_id, &task_id)? {
                    if task.status == ProspectiveTaskStatus::Pending {
                        due_tasks.push((user_id, task));
                    }
                }
            }
        }

        Ok(due_tasks)
    }

    /// Check for context-triggered reminders based on text content (keyword match only)
    ///
    /// Returns tasks where:
    /// - Trigger is OnContext
    /// - Any keyword matches the context text
    /// - Status is Pending
    ///
    /// For semantic matching, use `check_context_triggers_semantic` instead.
    pub fn check_context_triggers(
        &self,
        user_id: &str,
        context: &str,
    ) -> Result<Vec<ProspectiveTask>> {
        let context_lower = context.to_lowercase();
        let mut matches = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        // Get all pending context-based tasks for user
        let pending_tasks = self.list_for_user(user_id, Some(ProspectiveTaskStatus::Pending))?;

        for task in pending_tasks {
            if seen_ids.contains(&task.id.0) {
                continue;
            }

            if let ProspectiveTrigger::OnContext { ref keywords, .. } = task.trigger {
                // Check if any keyword matches
                let matched = keywords
                    .iter()
                    .any(|kw| context_lower.contains(&kw.to_lowercase()));
                if matched {
                    seen_ids.insert(task.id.0);
                    matches.push(task);
                }
            }
        }

        // Sort by priority
        matches.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(matches)
    }

    /// Check for context-triggered reminders using both keyword AND semantic matching
    ///
    /// Returns tasks that either:
    /// - Have keyword matches in the context (score = 1.0), OR
    /// - Have semantic similarity above their threshold
    ///
    /// # Arguments
    /// * `user_id` - User to check reminders for
    /// * `context` - Current context text (for keyword matching)
    /// * `context_embedding` - Precomputed embedding of the context
    /// * `embed_fn` - Closure to compute embedding for task content
    ///
    /// # Returns
    /// Vector of (task, score) tuples sorted by score (highest first)
    pub fn check_context_triggers_semantic<F>(
        &self,
        user_id: &str,
        context: &str,
        context_embedding: &[f32],
        embed_fn: F,
    ) -> Result<Vec<(ProspectiveTask, f32)>>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        let context_lower = context.to_lowercase();
        let mut matches: Vec<(ProspectiveTask, f32)> = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        // Get all pending context-based tasks for user
        let pending_tasks = self.list_for_user(user_id, Some(ProspectiveTaskStatus::Pending))?;

        for task in pending_tasks {
            if seen_ids.contains(&task.id.0) {
                continue;
            }

            if let ProspectiveTrigger::OnContext {
                ref keywords,
                threshold,
            } = task.trigger
            {
                // 1. Check keyword matches first (fast path)
                let keyword_match = keywords
                    .iter()
                    .any(|kw| context_lower.contains(&kw.to_lowercase()));

                if keyword_match {
                    seen_ids.insert(task.id.0);
                    matches.push((task, 1.0)); // Perfect score for keyword match
                    continue;
                }

                // 2. Try semantic matching (prefer cached embedding, fallback to embed_fn)
                let task_emb = task
                    .embedding
                    .as_deref()
                    .map(std::borrow::Cow::Borrowed)
                    .or_else(|| embed_fn(&task.content).map(std::borrow::Cow::Owned));

                if let Some(task_embedding) = task_emb {
                    let similarity = cosine_similarity(context_embedding, &task_embedding);
                    if similarity >= threshold {
                        seen_ids.insert(task.id.0);
                        matches.push((task, similarity));
                    }
                }
            }
        }

        // Sort by score (highest first), then by priority
        matches.sort_by(|a, b| {
            let score_cmp = b.1.total_cmp(&a.1);
            if score_cmp != std::cmp::Ordering::Equal {
                return score_cmp;
            }
            b.0.priority.cmp(&a.0.priority)
        });

        Ok(matches)
    }

    /// Mark a task as triggered
    pub fn mark_triggered(&self, user_id: &str, task_id: &ProspectiveTaskId) -> Result<bool> {
        if let Some(mut task) = self.get(user_id, task_id)? {
            if task.status == ProspectiveTaskStatus::Pending {
                task.mark_triggered();
                self.update(&task)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Mark a task as dismissed
    pub fn mark_dismissed(&self, user_id: &str, task_id: &ProspectiveTaskId) -> Result<bool> {
        if let Some(mut task) = self.get(user_id, task_id)? {
            task.mark_dismissed();
            self.update(&task)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Get count of pending tasks for a user
    pub fn pending_count(&self, user_id: &str) -> Result<usize> {
        let tasks = self.list_for_user(user_id, Some(ProspectiveTaskStatus::Pending))?;
        Ok(tasks.len())
    }

    /// Find a task by ID prefix (for short ID lookups)
    ///
    /// Allows users to dismiss reminders using short IDs like "d8cdc580"
    /// instead of full UUIDs like "d8cdc580-bf96-403a-85c5-57098c7b1786"
    pub fn find_by_prefix(
        &self,
        user_id: &str,
        id_prefix: &str,
    ) -> Result<Option<ProspectiveTask>> {
        let prefix_lower = id_prefix.to_lowercase();
        let tasks = self.list_for_user(user_id, None)?;

        let matches: Vec<_> = tasks
            .into_iter()
            .filter(|t| t.id.0.to_string().to_lowercase().starts_with(&prefix_lower))
            .collect();

        match matches.len() {
            0 => Ok(None),
            1 => Ok(Some(matches.into_iter().next().unwrap())),
            _ => {
                // Multiple matches - return the first one but log warning
                tracing::warn!(
                    user_id = %user_id,
                    prefix = %id_prefix,
                    matches = matches.len(),
                    "Multiple reminders match prefix, using first"
                );
                Ok(Some(matches.into_iter().next().unwrap()))
            }
        }
    }

    /// Cleanup expired/dismissed tasks older than given days
    pub fn cleanup_old_tasks(&self, user_id: &str, older_than_days: i64) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::days(older_than_days);
        let mut deleted = 0;

        let tasks = self.list_for_user(user_id, None)?;
        for task in tasks {
            // Only cleanup dismissed or expired tasks
            if matches!(
                task.status,
                ProspectiveTaskStatus::Dismissed | ProspectiveTaskStatus::Expired
            ) {
                let check_time = task.dismissed_at.unwrap_or(task.created_at);
                if check_time < cutoff {
                    self.delete(user_id, &task.id)?;
                    deleted += 1;
                }
            }
        }

        if deleted > 0 {
            tracing::info!(user_id = %user_id, deleted = deleted, "Cleaned up old prospective tasks");
        }

        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_test_db(path: &Path) -> Arc<DB> {
        let shared_path = path.join("shared");
        std::fs::create_dir_all(&shared_path).unwrap();
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        let cfs = vec![
            ColumnFamilyDescriptor::new("default", opts.clone()),
            ColumnFamilyDescriptor::new(CF_PROSPECTIVE, opts.clone()),
            ColumnFamilyDescriptor::new(CF_PROSPECTIVE_INDEX, opts.clone()),
        ];
        Arc::new(DB::open_cf_descriptors(&opts, &shared_path, cfs).unwrap())
    }

    fn setup_store() -> (TempDir, ProspectiveStore) {
        let temp_dir = TempDir::new().unwrap();
        let db = open_test_db(temp_dir.path());
        let store = ProspectiveStore::new(db, temp_dir.path()).unwrap();
        (temp_dir, store)
    }

    #[test]
    fn test_with_store_redb_round_trip() {
        // The point of with_store: the same store logic over the other backend,
        // including the due-index scan that leans on the ascending-order contract.
        let temp_dir = TempDir::new().unwrap();
        let redb_db = crate::storage::redb::create_redb_database_for_test(
            &temp_dir.path().join("prospective.redb"),
        )
        .unwrap();
        let store = ProspectiveStore::with_store(Arc::new(
            crate::storage::redb::RedbColumnStore::new(redb_db),
        ))
        .unwrap();

        let task = ProspectiveTask::new(
            "u".to_string(),
            "remember".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 1,
                from: Utc::now() - chrono::Duration::hours(1),
            },
        );
        store.store(&task).unwrap();
        assert_eq!(store.get("u", &task.id).unwrap().unwrap().content, "remember");

        // Past-due pending task must surface through the ordered due scan.
        let due = store.get_all_due_tasks().unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].0, "u");

        assert!(store.mark_dismissed("u", &task.id).unwrap());
        assert!(store.delete("u", &task.id).unwrap());
        assert!(store.get("u", &task.id).unwrap().is_none());
        assert!(store.list_for_user("u", None).unwrap().is_empty());
        // No engine DB behind this store — the backup accessor must say so.
        assert!(store.databases().is_empty());
    }

    #[test]
    fn test_store_and_get() {
        let (_temp, store) = setup_store();

        let task = ProspectiveTask::new(
            "test-user".to_string(),
            "Remember to push code".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 3600,
                from: Utc::now(),
            },
        );

        store.store(&task).unwrap();

        let retrieved = store.get("test-user", &task.id).unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.content, "Remember to push code");
        assert_eq!(retrieved.status, ProspectiveTaskStatus::Pending);
    }

    #[test]
    fn test_list_for_user() {
        let (_temp, store) = setup_store();

        // Create tasks for two users
        let task1 = ProspectiveTask::new(
            "user-a".to_string(),
            "Task 1".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 3600,
                from: Utc::now(),
            },
        );

        let task2 = ProspectiveTask::new(
            "user-a".to_string(),
            "Task 2".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 7200,
                from: Utc::now(),
            },
        );

        let task3 = ProspectiveTask::new(
            "user-b".to_string(),
            "Task 3".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 3600,
                from: Utc::now(),
            },
        );

        store.store(&task1).unwrap();
        store.store(&task2).unwrap();
        store.store(&task3).unwrap();

        let user_a_tasks = store.list_for_user("user-a", None).unwrap();
        assert_eq!(user_a_tasks.len(), 2);

        let user_b_tasks = store.list_for_user("user-b", None).unwrap();
        assert_eq!(user_b_tasks.len(), 1);
    }

    #[test]
    fn test_due_tasks() {
        let (_temp, store) = setup_store();

        // Task that's already due (0 seconds from past)
        let past = Utc::now() - chrono::Duration::seconds(100);
        let due_task = ProspectiveTask::new(
            "test-user".to_string(),
            "Due task".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 0,
                from: past,
            },
        );

        // Task that's not due yet
        let future_task = ProspectiveTask::new(
            "test-user".to_string(),
            "Future task".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 999999,
                from: Utc::now(),
            },
        );

        store.store(&due_task).unwrap();
        store.store(&future_task).unwrap();

        let due = store.get_due_tasks("test-user").unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].content, "Due task");
    }

    #[test]
    fn test_context_trigger() {
        let (_temp, store) = setup_store();

        let task = ProspectiveTask::new(
            "test-user".to_string(),
            "Check auth token".to_string(),
            ProspectiveTrigger::OnContext {
                keywords: vec![
                    "authentication".to_string(),
                    "token".to_string(),
                    "jwt".to_string(),
                ],
                threshold: 0.7,
            },
        );

        store.store(&task).unwrap();

        // Should match on keyword
        let matches = store
            .check_context_triggers("test-user", "I need to fix the JWT token expiry")
            .unwrap();
        assert_eq!(matches.len(), 1);

        // Should not match
        let no_matches = store
            .check_context_triggers("test-user", "Let's update the database schema")
            .unwrap();
        assert_eq!(no_matches.len(), 0);
    }

    #[test]
    fn test_due_key_migration_and_ordering() {
        let temp_dir = TempDir::new().unwrap();
        let db = open_test_db(temp_dir.path());

        let index_cf = db
            .cf_handle(CF_PROSPECTIVE_INDEX)
            .expect("prospective_index CF must exist");

        // Write unpadded keys simulating old format
        let task_id_a = uuid::Uuid::new_v4();
        let task_id_b = uuid::Uuid::new_v4();
        // ts_a = 9 (1 digit), ts_b = 10 (2 digits)
        // Without padding: "due:9:..." > "due:10:..." lexicographically (wrong)
        db.put_cf(
            index_cf,
            format!("due:9:{}", task_id_a).as_bytes(),
            b"user-1",
        )
        .unwrap();
        db.put_cf(
            index_cf,
            format!("due:10:{}", task_id_b).as_bytes(),
            b"user-1",
        )
        .unwrap();

        // Run migration through the trait, as new() does
        let store = crate::storage::legacy_rocksdb::RocksDbColumnStore::new(Arc::clone(&db));
        let migrated = migrate_due_key_padding(&store).unwrap();
        assert_eq!(migrated, 2);

        // Verify old keys are gone
        assert!(db
            .get_cf(index_cf, format!("due:9:{}", task_id_a).as_bytes())
            .unwrap()
            .is_none());
        assert!(db
            .get_cf(index_cf, format!("due:10:{}", task_id_b).as_bytes())
            .unwrap()
            .is_none());

        // Verify new padded keys exist
        let key_a = format!("due:{:020}:{}", 9_i64, task_id_a);
        let key_b = format!("due:{:020}:{}", 10_i64, task_id_b);
        assert!(db.get_cf(index_cf, key_a.as_bytes()).unwrap().is_some());
        assert!(db.get_cf(index_cf, key_b.as_bytes()).unwrap().is_some());

        // Verify lexicographic order is now correct: 9 < 10
        assert!(
            key_a < key_b,
            "Padded key for ts=9 should sort before ts=10"
        );

        // Re-running migration should be a no-op
        let migrated_again = migrate_due_key_padding(&store).unwrap();
        assert_eq!(migrated_again, 0);
    }

    #[test]
    fn test_mark_triggered_and_dismissed() {
        let (_temp, store) = setup_store();

        let task = ProspectiveTask::new(
            "test-user".to_string(),
            "Test task".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 0,
                from: Utc::now(),
            },
        );

        store.store(&task).unwrap();

        // Mark as triggered
        store.mark_triggered("test-user", &task.id).unwrap();
        let updated = store.get("test-user", &task.id).unwrap().unwrap();
        assert_eq!(updated.status, ProspectiveTaskStatus::Triggered);
        assert!(updated.triggered_at.is_some());

        // Mark as dismissed
        store.mark_dismissed("test-user", &task.id).unwrap();
        let dismissed = store.get("test-user", &task.id).unwrap().unwrap();
        assert_eq!(dismissed.status, ProspectiveTaskStatus::Dismissed);
        assert!(dismissed.dismissed_at.is_some());
    }

    #[test]
    fn test_delete() {
        let (_temp, store) = setup_store();

        let task = ProspectiveTask::new(
            "test-user".to_string(),
            "To delete".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 3600,
                from: Utc::now(),
            },
        );

        store.store(&task).unwrap();
        assert!(store.get("test-user", &task.id).unwrap().is_some());

        let deleted = store.delete("test-user", &task.id).unwrap();
        assert!(deleted);
        assert!(store.get("test-user", &task.id).unwrap().is_none());
    }
}
