//! Backups on the redb backend (ADR 0006 WP3): a WP2 export plus the intent log, a manifest, and a rename into place.

use super::*;
use crate::backup::{BackupMetadata, BackupType, IntentLogBackupSpec, IntentLogFileMeta};
use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

pub const BACKUP_FORMAT: &str = "veld-redb-backup-1";
const BACKUP_MANIFEST: &str = "backup.json";
const EXPORT_DIR: &str = "export";
const INTENT_DIR: &str = "intent_log";
const JOURNAL: &str = "intent.log";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitRecord {
    pub unit: Unit,
    pub file: String,
    pub frames: u64,
    pub records: u64,
    pub digest: String,
    pub keyspaces: BTreeMap<String, u64>,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    pub name: String,
    pub captured: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    pub format: String,
    pub backup_id: u32,
    pub created_at: DateTime<Utc>,
    pub backend: String,
    pub veld_version: String,
    pub user_id: String,
    pub units: Vec<UnitRecord>,
    pub intent_log_files: Vec<IntentLogFileMeta>,
    pub journal_cut_lsn: Option<u64>,
    pub components: Vec<Component>,
    pub metadata: BackupMetadata,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerifyReport {
    pub backup_id: u32,
    pub units: Vec<UnitRecord>,
    pub intent_log_files: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RestoreReport {
    pub backup_id: u32,
    pub imported: Vec<ImportReport>,
    pub intent_log_files: Vec<String>,
    pub not_restored: Vec<String>,
}

pub struct RedbBackupEngine {
    root: PathBuf,
    lock: parking_lot::Mutex<()>,
}

impl RedbBackupEngine {
    pub fn new(root: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&root)
            .with_context(|| format!("creating backup root {}", root.display()))?;
        Ok(Self {
            root,
            lock: parking_lot::Mutex::new(()),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn user_root(&self, user_id: &str) -> Result<PathBuf> {
        Ok(self.root.join(crate::path_guard::sanitize_component(
            user_id,
            "backup user id",
        )?))
    }

    fn backup_dir(&self, user_id: &str, backup_id: u32) -> Result<PathBuf> {
        Ok(self.user_root(user_id)?.join(backup_id.to_string()))
    }

    fn ids(&self, user_id: &str) -> Result<Vec<u32>> {
        let dir = self.user_root(user_id)?;
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
        };
        let mut ids = Vec::new();
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let Some(id) = name.to_str().and_then(|n| n.parse::<u32>().ok()) else {
                continue;
            };
            if entry.file_type()?.is_dir() {
                ids.push(id);
            }
        }
        ids.sort_unstable();
        Ok(ids)
    }

    fn load_manifest(&self, user_id: &str, backup_id: u32) -> Result<BackupManifest> {
        let path = self.backup_dir(user_id, backup_id)?.join(BACKUP_MANIFEST);
        let bytes = std::fs::read(&path).with_context(|| {
            format!(
                "no readable backup {backup_id} for '{user_id}' at {}",
                path.display()
            )
        })?;
        let manifest: BackupManifest = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        if manifest.format != BACKUP_FORMAT
            || manifest.backend != StorageBackend::Redb.to_string()
            || manifest.backup_id != backup_id
            || manifest.user_id != user_id
        {
            bail!(
                "{} describes {} backup {} of '{}' on {}, not {BACKUP_FORMAT} backup {backup_id} of '{user_id}' on redb",
                path.display(),
                manifest.format,
                manifest.backup_id,
                manifest.user_id,
                manifest.backend
            );
        }
        Ok(manifest)
    }

    /// Holds the engine lock until the backup is finished or dropped, so ids are never shared.
    pub fn begin(&self, user_id: &str) -> Result<PendingBackup<'_>> {
        let guard = self.lock.lock();
        let user_root = self.user_root(user_id)?;
        std::fs::create_dir_all(&user_root)?;
        let backup_id = self.ids(user_id)?.last().copied().unwrap_or(0) + 1;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = user_root.join(format!(
            ".partial-{backup_id}-{}-{nanos}",
            std::process::id()
        ));
        prepare_private_dir(&tmp, "backup build directory")?;
        Ok(PendingBackup {
            _guard: guard,
            user_id: user_id.to_string(),
            backup_id,
            created_at: Utc::now(),
            final_dir: user_root.join(backup_id.to_string()),
            tmp,
            intent: None,
            units: None,
            finished: false,
        })
    }

    pub fn create(
        &self,
        base: &Path,
        user_id: &str,
        live: &LiveRedb,
        intent_log: &IntentLogBackupSpec,
    ) -> Result<BackupMetadata> {
        let mut pending = self.begin(user_id)?;
        pending.stage_intent_log(intent_log)?;
        pending.export(base, live)?;
        pending.finish()
    }

    pub fn list(&self, user_id: &str) -> Result<Vec<BackupMetadata>> {
        let mut out = Vec::new();
        for id in self.ids(user_id)?.into_iter().rev() {
            out.push(self.load_manifest(user_id, id)?.metadata);
        }
        Ok(out)
    }

    /// `Ok` only when every dump and intent-log file re-derives exactly what the manifest recorded.
    pub fn verify(&self, user_id: &str, backup_id: u32) -> Result<VerifyReport> {
        let manifest = self.load_manifest(user_id, backup_id)?;
        let dir = self.backup_dir(user_id, backup_id)?;
        let export_dir = dir.join(EXPORT_DIR);

        let mut on_disk = collect_dumps(&export_dir)
            .with_context(|| format!("backup {backup_id} of '{user_id}': export"))?;
        on_disk.sort();
        let mut expected: Vec<PathBuf> = manifest
            .units
            .iter()
            .map(|u| export_dir.join(&u.file))
            .collect();
        expected.sort();
        if on_disk != expected {
            bail!(
                "backup {backup_id} of '{user_id}': export holds {on_disk:?}, the backup manifest lists {expected:?}"
            );
        }
        for recorded in &manifest.units {
            let derived = summarise_dump(&export_dir.join(&recorded.file), &recorded.file)
                .with_context(|| format!("backup {backup_id} of '{user_id}': {}", recorded.file))?;
            if &derived != recorded {
                bail!(
                    "backup {backup_id} of '{user_id}': {} re-derives {derived:?}, the manifest recorded {recorded:?}",
                    recorded.file
                );
            }
        }

        let intent_dir = dir.join(INTENT_DIR);
        let mut present = Vec::new();
        match std::fs::read_dir(&intent_dir) {
            Ok(entries) => {
                for entry in entries {
                    present.push(entry?.file_name().to_string_lossy().into_owned());
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("reading {}", intent_dir.display())),
        }
        present.sort();
        let mut listed: Vec<String> = manifest
            .intent_log_files
            .iter()
            .map(|f| f.name.clone())
            .collect();
        listed.sort();
        if present != listed {
            bail!(
                "backup {backup_id} of '{user_id}': intent_log holds {present:?}, the manifest lists {listed:?}"
            );
        }
        for file in &manifest.intent_log_files {
            let derived =
                hash_intent_file(&intent_dir.join(&file.name), &file.name).with_context(|| {
                    format!(
                        "backup {backup_id} of '{user_id}': intent_log/{}",
                        file.name
                    )
                })?;
            if (&derived.sha256, derived.size_bytes, derived.lsn_range)
                != (&file.sha256, file.size_bytes, file.lsn_range)
            {
                bail!(
                    "backup {backup_id} of '{user_id}': intent_log/{} re-derives sha256 {} ({} bytes, lsn {:?}), \
                     the manifest recorded {} ({} bytes, lsn {:?})",
                    file.name,
                    derived.sha256,
                    derived.size_bytes,
                    derived.lsn_range,
                    file.sha256,
                    file.size_bytes,
                    file.lsn_range
                );
            }
        }

        let checksum = content_checksum(&manifest.units, &manifest.intent_log_files);
        if checksum != manifest.metadata.checksum {
            bail!(
                "backup {backup_id} of '{user_id}': content checksum {checksum} does not match the recorded {}",
                manifest.metadata.checksum
            );
        }
        Ok(VerifyReport {
            backup_id,
            units: manifest.units,
            intent_log_files: manifest.intent_log_files.len(),
        })
    }

    pub fn purge(&self, user_id: &str, keep_count: usize) -> Result<usize> {
        if keep_count == 0 {
            bail!("keep_count must be >= 1 to prevent deleting all backups");
        }
        let _guard = self.lock.lock();
        let ids = self.ids(user_id)?;
        if ids.len() <= keep_count {
            return Ok(0);
        }
        let doomed = &ids[..ids.len() - keep_count];
        let user_root = self.user_root(user_id)?;
        for id in doomed {
            // Renamed first so a half-removed backup is never listed as a broken one.
            let purging = user_root.join(format!(".purging-{id}"));
            std::fs::rename(user_root.join(id.to_string()), &purging)
                .with_context(|| format!("setting backup {id} of '{user_id}' aside for purge"))?;
            std::fs::remove_dir_all(&purging)
                .with_context(|| format!("removing {}", purging.display()))?;
        }
        tracing::info!(
            user_id,
            purged = doomed.len(),
            kept = keep_count,
            "redb backups purged"
        );
        Ok(doomed.len())
    }

    /// Without `include_shared` the shared stores are reported, not restored: they hold every tenant's rows.
    pub fn restore_into(
        &self,
        user_id: &str,
        backup_id: u32,
        target_base: &Path,
        options: &crate::backup::RestoreOptions,
        include_shared: bool,
    ) -> Result<RestoreReport> {
        let verified = self.verify(user_id, backup_id)?;
        let manifest = self.load_manifest(user_id, backup_id)?;
        let dir = self.backup_dir(user_id, backup_id)?;
        let export_dir = dir.join(EXPORT_DIR);
        let target_user = user_dir(target_base, user_id)?;

        if options.max_lsn.is_some() && !manifest.intent_log_files.iter().any(|f| f.name == JOURNAL)
        {
            bail!("max_lsn requested but backup {backup_id} of '{user_id}' holds no journal");
        }
        for file in &manifest.intent_log_files {
            let target = target_user.join(&file.name);
            if exists(&target)? {
                bail!(
                    "{} already exists — refusing to restore over it",
                    target.display()
                );
            }
        }

        let input = if include_shared {
            export_dir.clone()
        } else {
            let user_unit = Unit::User(user_id.to_string());
            let record = verified
                .units
                .iter()
                .find(|u| u.unit == user_unit)
                .ok_or_else(|| {
                    anyhow!("backup {backup_id} of '{user_id}' holds no dump for {user_unit}")
                })?;
            export_dir.join(&record.file)
        };
        let imported = import(target_base, StorageBackend::Redb, &input)?;

        let mut restored_files = Vec::new();
        for file in &manifest.intent_log_files {
            let target = target_user.join(&file.name);
            let (sha256, size_bytes) =
                crate::backup::copy_and_hash(&dir.join(INTENT_DIR).join(&file.name), &target)?;
            if sha256 != file.sha256 || size_bytes != file.size_bytes {
                bail!(
                    "intent_log/{} changed between verification and restore (sha256 {sha256}, expected {})",
                    file.name,
                    file.sha256
                );
            }
            restored_files.push(file.name.clone());
        }
        let journal = target_user.join(JOURNAL);
        if exists(&journal)? {
            crate::backup::scan_lsn_range(&journal)
                .with_context(|| format!("restored journal {}", journal.display()))?;
            if let Some(max_lsn) = options.max_lsn {
                let mut log = crate::intent_log::IntentLog::open(&journal)?;
                log.truncate_to_lsn(crate::intent_log::Lsn(max_lsn))
                    .map_err(|e| anyhow!("PITR truncate_to_lsn({max_lsn}) failed: {e}"))?;
                log.sync().context("syncing the truncated intent log")?;
            }
        }

        let not_restored = if include_shared {
            Vec::new()
        } else {
            vec![format!(
                "shared stores and audit (captured in backup {backup_id}, not restored: they hold \
                 every tenant's rows; import {} into an empty data directory to recover them)",
                export_dir.join(dump_file_name(&Unit::Shared)).display()
            )]
        };
        Ok(RestoreReport {
            backup_id,
            imported,
            intent_log_files: restored_files,
            not_restored,
        })
    }
}

pub struct PendingBackup<'e> {
    _guard: parking_lot::MutexGuard<'e, ()>,
    user_id: String,
    backup_id: u32,
    created_at: DateTime<Utc>,
    tmp: PathBuf,
    final_dir: PathBuf,
    intent: Option<(Vec<IntentLogFileMeta>, Option<u64>)>,
    units: Option<Vec<UnitRecord>>,
    finished: bool,
}

impl PendingBackup<'_> {
    pub fn backup_id(&self) -> u32 {
        self.backup_id
    }

    /// Call before `export`, under the journal writer, so the stores are a superset of the staged journal.
    pub fn stage_intent_log(&mut self, spec: &IntentLogBackupSpec) -> Result<()> {
        let files = crate::backup::copy_intent_log_files(spec, &self.tmp.join(INTENT_DIR))?;
        let cut = files
            .iter()
            .find(|f| f.name == spec.log_filename)
            .and_then(|f| f.lsn_range)
            .map(|(_, hi)| hi);
        self.intent = Some((files, cut));
        Ok(())
    }

    pub fn export(&mut self, base: &Path, live: &LiveRedb) -> Result<()> {
        let units = [Unit::User(self.user_id.clone()), Unit::Shared];
        let export_dir = self.tmp.join(EXPORT_DIR);
        let reports = export_live(base, &units, live, &export_dir)?;
        let mut records = Vec::new();
        for report in &reports {
            let file = dump_file_name(&report.unit);
            let summary = summarise_dump(&report.path, &file)?;
            if summary.unit != report.unit
                || summary.frames != report.frames
                || summary.digest != report.digest
            {
                bail!(
                    "{}: reads back as {} frames ({}) for {}, was written as {} frames ({}) for {}",
                    report.path.display(),
                    summary.frames,
                    summary.digest,
                    summary.unit,
                    report.frames,
                    report.digest,
                    report.unit
                );
            }
            records.push(summary);
        }
        self.units = Some(records);
        Ok(())
    }

    pub fn finish(mut self) -> Result<BackupMetadata> {
        let units = self.units.clone().ok_or_else(|| {
            anyhow!(
                "backup {} has no export — refusing to finish it",
                self.backup_id
            )
        })?;
        let (intent_files, cut) = self.intent.clone().ok_or_else(|| {
            anyhow!(
                "backup {} did not stage the intent log — refusing to finish it without one",
                self.backup_id
            )
        })?;
        let user_unit = Unit::User(self.user_id.clone());
        let user_record = units
            .iter()
            .find(|u| u.unit == user_unit)
            .ok_or_else(|| anyhow!("backup {} holds no dump for {user_unit}", self.backup_id))?;
        let shared_record = units
            .iter()
            .find(|u| u.unit == Unit::Shared)
            .ok_or_else(|| anyhow!("backup {} holds no shared dump", self.backup_id))?;
        let prefixed = |record: &UnitRecord, prefix: &str| -> Vec<String> {
            record
                .keyspaces
                .keys()
                .filter_map(|k| k.strip_prefix(prefix).map(str::to_string))
                .collect()
        };
        let primary_tables = prefixed(user_record, "primary/");
        let graph_tables = prefixed(user_record, "graph/");
        let shared_tables = prefixed(shared_record, "shared/");
        let audit_rows = shared_record
            .keyspaces
            .get("audit/audit")
            .copied()
            .unwrap_or(0);

        let components = vec![
            Component {
                name: "primary".into(),
                captured: true,
                detail: format!("storage.redb tables {primary_tables:?}"),
            },
            Component {
                name: "graph".into(),
                captured: true,
                detail: format!("graph/graph.redb tables {graph_tables:?}"),
            },
            Component {
                name: "shared".into(),
                captured: true,
                detail: format!("shared.redb tables {shared_tables:?}"),
            },
            Component {
                name: "audit".into(),
                captured: true,
                detail: format!("audit.redb, {audit_rows} rows"),
            },
            Component {
                name: "intent_log".into(),
                captured: true,
                detail: format!("{} file(s), journal cut {cut:?}", intent_files.len()),
            },
            Component {
                name: "memory_index".into(),
                captured: false,
                detail: "rocksdb-only derived index; redb keeps none".into(),
            },
            Component {
                name: "user_auth".into(),
                captured: false,
                detail: "user-auth requires the rocksdb engine and has no redb store".into(),
            },
            Component {
                name: "projections".into(),
                captured: false,
                detail: "vector, BM25 and SQLite projections are rebuilt from the intent log; \
                         the rocksdb backup excludes them too"
                    .into(),
            },
        ];

        let size_bytes = units.iter().map(|u| u.size_bytes).sum::<u64>()
            + intent_files.iter().map(|f| f.size_bytes).sum::<u64>();
        let metadata = BackupMetadata {
            backup_id: self.backup_id,
            created_at: self.created_at,
            user_id: self.user_id.clone(),
            backup_type: BackupType::Full,
            size_bytes,
            checksum: content_checksum(&units, &intent_files),
            memory_count: user_record
                .keyspaces
                .get("primary/memories")
                .copied()
                .unwrap_or(0) as usize,
            // redb has no sequence number; 0 here means absent, not a PITR anchor.
            sequence_number: 0,
            secondary_stores: vec!["audit".into(), "graph".into(), "shared".into()],
            secondary_size_bytes: shared_record.size_bytes,
            intent_log_files: intent_files.clone(),
            journal_cut_lsn: cut,
            column_families: primary_tables,
        };
        let manifest = BackupManifest {
            format: BACKUP_FORMAT.to_string(),
            backup_id: self.backup_id,
            created_at: self.created_at,
            backend: StorageBackend::Redb.to_string(),
            veld_version: env!("CARGO_PKG_VERSION").to_string(),
            user_id: self.user_id.clone(),
            units,
            intent_log_files: intent_files,
            journal_cut_lsn: cut,
            components,
            metadata: metadata.clone(),
        };
        write_atomic(
            &self.tmp.join(BACKUP_MANIFEST),
            &serde_json::to_vec_pretty(&manifest)?,
        )?;
        if exists(&self.final_dir)? {
            bail!(
                "{} already exists — refusing to replace a backup",
                self.final_dir.display()
            );
        }
        std::fs::rename(&self.tmp, &self.final_dir).with_context(|| {
            format!(
                "renaming {} into place as {}",
                self.tmp.display(),
                self.final_dir.display()
            )
        })?;
        self.finished = true;
        tracing::info!(
            user_id = %self.user_id,
            backup_id = self.backup_id,
            memory_count = metadata.memory_count,
            "redb backup written"
        );
        Ok(metadata)
    }
}

impl Drop for PendingBackup<'_> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = std::fs::remove_dir_all(&self.tmp);
        }
    }
}

fn summarise_dump(path: &Path, file: &str) -> Result<UnitRecord> {
    let records = portable::read_all(
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?,
    )
    .with_context(|| format!("reading {}", path.display()))?;
    let (frames, digest) = portable::read_trailer(path)?;
    let (unit, data) = parse_dump(records)?;
    let mut keyspaces = BTreeMap::new();
    for rec in &data {
        *keyspaces.entry(rec.cf.clone()).or_insert(0u64) += 1;
    }
    Ok(UnitRecord {
        unit,
        file: file.to_string(),
        frames,
        records: data.len() as u64,
        digest: portable::hex(&digest),
        keyspaces,
        size_bytes: std::fs::metadata(path)?.len(),
    })
}

fn hash_intent_file(path: &Path, name: &str) -> Result<IntentLogFileMeta> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening intent log file {}", path.display()))?;
    let mut hasher = Sha256::new();
    let size_bytes = std::io::copy(&mut file, &mut hasher)?;
    let lsn_range = if name == JOURNAL {
        crate::backup::scan_lsn_range(path)?
    } else {
        None
    };
    Ok(IntentLogFileMeta {
        name: name.to_string(),
        sha256: format!("{:x}", hasher.finalize()),
        size_bytes,
        lsn_range,
    })
}

fn content_checksum(units: &[UnitRecord], intent_files: &[IntentLogFileMeta]) -> String {
    let mut hasher = Sha256::new();
    for u in units {
        hasher.update(format!("dump\t{}\t{}\t{}\n", u.file, u.frames, u.digest));
    }
    for f in intent_files {
        hasher.update(format!(
            "intent\t{}\t{}\t{}\n",
            f.name, f.size_bytes, f.sha256
        ));
    }
    format!("{:x}", hasher.finalize())
}

pub use crate::backup::swap_files_into_place;

#[cfg(test)]
mod tests;
