//! Backend-to-backend migration over the portable dump — ADR 0006 WP2.
//!
//! The two engines do not share a physical layout for the primary memory store:
//! rocksdb keeps memories, vector mappings and a dozen prefixed subsystems in one
//! default column family plus a derived `memory_index`, while redb splits them into
//! `memories`, `vector_mappings` and `default` tables beside its derived `index_*`
//! tables. A dump of
//! one engine's physical rows imports into the other as tables nothing reads — a
//! store that verifies and serves zero records, which is the outage 0006 records.
//!
//! So a migration dump is LOGICAL. Every frame is named `<component>/<keyspace>`:
//!
//! | keyspace                  | rocksdb                                  | redb                               |
//! |---------------------------|------------------------------------------|------------------------------------|
//! | `primary/memories`        | `<u>/storage` default, 16-byte keys      | `<u>/storage.redb` `memories`      |
//! | `primary/vector_mappings` | default, `vmapping:{uuid}`               | `vector_mappings`, uuid bytes      |
//! | `primary/default`         | every other default-family row           | `default`                          |
//! | `primary/kv`              | `kv`                                     | `kv`                               |
//! | `graph/<family>`          | `<u>/graph/graph` family                 | `<u>/graph/graph.redb` table       |
//! | `audit/audit`             | `shared` family `audit`                  | `audit.redb` `audit`               |
//! | `shared/<family>`         | `shared` family                          | `shared.redb` table                |
//!
//! `memory_index` and redb's `index_*` tables are derived and are not carried: they are
//! counted, and rebuilt from the memories on the target — by `rebuild_index` on rocksdb,
//! by the first open on redb. Keyspaces are emitted in name order and keys in byte
//! order, so an extraction of the target after an import reproduces the dump's frames
//! exactly — which is how an import proves itself against the trailer.
//!
//! A tenant with record-level encryption migrates sealed. Its keystore (`<u>/storage/
//! keystore.json` on rocksdb, `<u>/keystore.json` on redb) is copied to the target's
//! place; sealed memories travel as stored, each one opened under the keystore on the
//! way out and again on the target; and the keystore generation sentinel travels as
//! `primary/default` `meta:keystore_generation`, which rocksdb keeps in `memory_index`.

#[cfg(feature = "storage-redb")]
pub mod backup;
mod readonly;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::config::StorageBackend;
use crate::storage::portable::{self, PortableRecord, PortableWriter};

pub const FORMAT: &str = "veld-logical-1";
const MANIFEST: &str = "MANIFEST.json";
const META: &str = "meta";
const KEYSTORE: &str = "keystore.json";
#[cfg(feature = "rocksdb")]
const VMAPPING: &[u8] = b"vmapping:";
#[cfg(feature = "rocksdb")]
const ROCKS_BATCH_OPS: usize = 10_000;

/// What one dump holds: one user's per-user stores, or the stores every user shares.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    User(String),
    Shared,
}

impl std::fmt::Display for Unit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unit::User(id) => write!(f, "user '{id}'"),
            Unit::Shared => f.write_str("shared stores"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    User(String),
    /// Every user with a store on the source backend, plus the shared stores.
    All,
}

impl Scope {
    /// `all` is the keyword; any other value must be a valid user id.
    pub fn parse(value: &str) -> Result<Self> {
        if value == "all" {
            return Ok(Scope::All);
        }
        crate::path_guard::sanitize_component(value, "user id")?;
        Ok(Scope::User(value.to_string()))
    }
}

/// What an extraction saw. `frames` and `digest` are the trailer values the same rows
/// produce; `records` excludes the meta frames.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Extraction {
    pub frames: u64,
    pub records: u64,
    pub digest: String,
    /// Records no decoder accepts that were NOT set aside. Any non-zero value fails an export.
    pub decode_failures: u64,
    pub failed_keys: Vec<String>,
    /// Undecodable records written byte-for-byte to a quarantine file, never exported.
    pub quarantined: u64,
    pub quarantined_keys: Vec<String>,
    pub derived_rows_not_carried: u64,
    /// Sealed memories carried as stored, each shown to open under the keystore.
    #[serde(skip_serializing_if = "is_zero")]
    pub sealed_records: u64,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

impl Extraction {
    fn holds_anything(&self) -> bool {
        self.records > 0 || self.decode_failures > 0 || self.derived_rows_not_carried > 0
    }

    /// Every record the store holds that a migration is accountable for.
    pub fn source_records(&self) -> u64 {
        self.records + self.decode_failures + self.quarantined
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DumpReport {
    pub unit: Unit,
    pub path: PathBuf,
    pub frames: u64,
    pub records: u64,
    pub digest: String,
    pub derived_rows_not_carried: u64,
    /// Set aside, not exported. `records + quarantined == source_records`.
    pub quarantined: u64,
    pub source_records: u64,
    pub quarantine: Option<QuarantineReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub unit: Unit,
    pub source: PathBuf,
    pub frames: u64,
    pub records: u64,
    pub digest: String,
    /// Frames and digest re-derived from the target after the import. Equal to the
    /// trailer by construction of `Ok`; reported so the equality is visible.
    pub verified_frames: u64,
    pub verified_digest: String,
    /// Sealed memories on the target, each opened under the keystore copied there.
    #[serde(skip_serializing_if = "is_zero")]
    pub verified_sealed_records: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct UnitPlan {
    pub unit: Unit,
    pub source: Option<Extraction>,
    pub target: Option<Extraction>,
    pub blocked: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MigrateReport {
    pub from: String,
    pub to: String,
    pub dry_run: bool,
    pub quarantine_requested: bool,
    pub units: Vec<UnitPlan>,
    pub imported: Vec<ImportReport>,
    /// Records set aside and NOT migrated. Empty unless quarantine was requested.
    pub quarantined: Vec<QuarantineReport>,
}

impl MigrateReport {
    pub fn blocked(&self) -> bool {
        self.units.iter().any(|u| !u.blocked.is_empty())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    format: String,
    source_backend: String,
    dumps: Vec<ManifestEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ManifestEntry {
    file: String,
    unit: Unit,
    frames: u64,
    digest: String,
}

// ─── paths ──────────────────────────────────────────────────────────────────────

fn user_dir(base: &Path, id: &str) -> Result<PathBuf> {
    Ok(base.join(crate::path_guard::sanitize_component(id, "user id")?))
}

fn exists(path: &Path) -> Result<bool> {
    match std::fs::metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e).with_context(|| format!("inspecting {}", path.display())),
    }
}

fn unit_present(base: &Path, backend: StorageBackend, unit: &Unit) -> Result<bool> {
    let candidates: Vec<PathBuf> = match (backend, unit) {
        (StorageBackend::RocksDb, Unit::User(id)) => {
            let dir = user_dir(base, id)?;
            vec![dir.join("storage/CURRENT"), dir.join("graph/graph/CURRENT")]
        }
        (StorageBackend::Redb, Unit::User(id)) => {
            let dir = user_dir(base, id)?;
            vec![dir.join("storage.redb"), dir.join("graph/graph.redb")]
        }
        (StorageBackend::RocksDb, Unit::Shared) => vec![base.join("shared/CURRENT")],
        (StorageBackend::Redb, Unit::Shared) => {
            vec![base.join("shared.redb"), base.join("audit.redb")]
        }
    };
    for path in candidates {
        if exists(&path)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The units a scope names on `backend`. A named user with no store there is an
/// error: a typo would otherwise export a valid, empty dump.
pub fn resolve_units(base: &Path, backend: StorageBackend, scope: &Scope) -> Result<Vec<Unit>> {
    match scope {
        Scope::User(id) => {
            let unit = Unit::User(id.clone());
            if !unit_present(base, backend, &unit)? {
                bail!(
                    "no {backend} store for user '{id}' under {}",
                    base.display()
                );
            }
            Ok(vec![unit])
        }
        Scope::All => {
            let entries = std::fs::read_dir(base)
                .with_context(|| format!("reading data directory {}", base.display()))?;
            let mut units = Vec::new();
            for entry in entries {
                let entry = entry?;
                if !entry.file_type()?.is_dir() {
                    continue;
                }
                let name = entry.file_name();
                let Some(name) = name.to_str() else {
                    bail!(
                        "{} holds a directory whose name is not utf-8",
                        base.display()
                    );
                };
                let candidate = Unit::User(name.to_string());
                if crate::validation::validate_user_id(name).is_err() {
                    let dir = entry.path();
                    let holds_store = [
                        "storage/CURRENT",
                        "graph/graph/CURRENT",
                        "storage.redb",
                        "graph/graph.redb",
                    ]
                    .iter()
                    .map(|p| exists(&dir.join(p)))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .any(|b| b);
                    if holds_store {
                        bail!("directory '{name}' holds a store but is not a valid user id — refusing to skip it");
                    }
                    continue;
                }
                if unit_present(base, backend, &candidate)? {
                    units.push(candidate);
                }
            }
            units.sort();
            if unit_present(base, backend, &Unit::Shared)? {
                units.push(Unit::Shared);
            }
            if units.is_empty() {
                bail!("no {backend} stores under {}", base.display());
            }
            Ok(units)
        }
    }
}

// ─── extraction ─────────────────────────────────────────────────────────────────

/// Undecodable records set aside by `--quarantine-undecodable`: the raw key and value
/// in a VELDPORT file of their own, and a listing with each value's sha256.
pub struct QuarantineWriter {
    writer: PortableWriter<std::io::BufWriter<std::fs::File>>,
    listing: Vec<serde_json::Value>,
}

/// How an extraction treats memory records sealed by record-level encryption.
#[derive(Clone, Copy)]
enum Sealed<'k> {
    /// export and import: an encrypted rocksdb store is refused, and redb carries sealed
    /// records as stored for a backup that restores under the same keystore.
    AsStored,
    /// migrate: a sealed record is carried as stored once it opens under the keys;
    /// without keys it is undecodable.
    Opened(Option<Keys<'k>>),
}

#[derive(Clone, Copy)]
struct Keys<'k> {
    cryptors: &'k crate::keystore::RecordCryptors,
    /// Source side: the keystore's generation, carried as the sentinel once the store's
    /// own sentinel is shown not to be ahead of it. Target side, `None`: the sentinel as stored.
    generation: Option<u64>,
}

struct Sink<'q, W: Write> {
    writer: PortableWriter<W>,
    stats: Extraction,
    quarantine: Option<&'q mut QuarantineWriter>,
    sealed: Sealed<'q>,
}

impl<'q, W: Write> Sink<'q, W> {
    fn frame(&mut self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        self.writer.write_record(&PortableRecord {
            cf: cf.to_string(),
            key: key.to_vec(),
            value: value.to_vec(),
        })
    }

    fn put(&mut self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        self.frame(cf, key, value)?;
        self.stats.records += 1;
        Ok(())
    }

    fn fail(&mut self, keyspace: &str, key: &[u8], value: &[u8], label: String) -> Result<()> {
        use sha2::{Digest, Sha256};
        match self.quarantine.as_deref_mut() {
            None => {
                self.stats.decode_failures += 1;
                self.stats.failed_keys.push(label);
            }
            Some(q) => {
                q.writer.write_record(&PortableRecord {
                    cf: keyspace.to_string(),
                    key: key.to_vec(),
                    value: value.to_vec(),
                })?;
                let digest: [u8; 32] = Sha256::digest(value).into();
                q.listing.push(serde_json::json!({
                    "record": label,
                    "keyspace": keyspace,
                    "key_hex": key.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                    "value_len": value.len(),
                    "value_sha256": portable::hex(&digest),
                }));
                self.stats.quarantined += 1;
                self.stats.quarantined_keys.push(label);
            }
        }
        Ok(())
    }

    fn keys(&self) -> Option<Keys<'q>> {
        match self.sealed {
            Sealed::Opened(keys) => keys,
            Sealed::AsStored => None,
        }
    }

    fn opens_sealed(&self) -> bool {
        matches!(self.sealed, Sealed::Opened(_))
    }

    fn sealed_memory(&mut self, key: &[u8], value: &[u8], label: String) -> Result<()> {
        match open_sealed(self.keys(), key, value) {
            Ok(()) => {
                self.stats.sealed_records += 1;
                self.put("primary/memories", key, value)
            }
            Err(reason) => self.fail(
                "primary/memories",
                key,
                value,
                format!("{label} ({reason})"),
            ),
        }
    }

    /// A `primary/default` row, with the keystore generation sentinel spliced in at its key's place.
    fn put_default(
        &mut self,
        key: &[u8],
        value: &[u8],
        sentinel: &mut Option<Vec<u8>>,
    ) -> Result<()> {
        if sentinel.is_some() && key >= GENERATION_SENTINEL {
            self.flush_sentinel(sentinel)?;
            if key == GENERATION_SENTINEL {
                return Ok(());
            }
        }
        self.put("primary/default", key, value)
    }

    fn flush_sentinel(&mut self, sentinel: &mut Option<Vec<u8>>) -> Result<()> {
        match sentinel.take() {
            Some(value) => self.put("primary/default", GENERATION_SENTINEL, &value),
            None => Ok(()),
        }
    }
}

const GENERATION_SENTINEL: &[u8] = crate::keystore::GENERATION_SENTINEL_KEY;

/// The sentinel frame an encrypted store contributes, given the value it holds.
fn sentinel_frame(
    keys: Keys<'_>,
    stored: Option<Vec<u8>>,
    store: &Path,
) -> Result<Option<Vec<u8>>> {
    let Some(generation) = keys.generation else {
        return Ok(stored);
    };
    let seen = stored
        .as_deref()
        .and_then(|b| <[u8; 8]>::try_from(b).ok())
        .map(u64::from_le_bytes)
        .unwrap_or(0);
    crate::keystore::keystore_generation_sentinel(seen, generation)
        .with_context(|| format!("{} and its keystore", store.display()))?;
    Ok(Some(generation.to_le_bytes().to_vec()))
}

/// `Ok` when a sealed memory opens under `keys` to a memory in the current layout,
/// which both engines read; otherwise why not.
fn open_sealed(
    keys: Option<Keys<'_>>,
    key: &[u8],
    value: &[u8],
) -> std::result::Result<(), String> {
    let keys = keys.ok_or("sealed, and the store has no keystore")?;
    if key.len() != 16 {
        return Err("sealed under a key that is not a memory id".to_string());
    }
    let epoch = crate::keystore::record_epoch(value).ok_or("sealed record header is truncated")?;
    let cryptor = keys.cryptors.for_epoch(epoch).ok_or_else(|| {
        format!("sealed under epoch {epoch}, which the keystore holds no key for")
    })?;
    let plaintext = zeroize::Zeroizing::new(
        cryptor
            .decrypt_record(value, key)
            .map_err(|e| format!("does not open under the keystore: {e}"))?,
    );
    if !crate::memory::storage::is_current_memory_value(&plaintext) {
        return Err("opens, but not to a memory in the current layout".to_string());
    }
    Ok(())
}

fn meta_frames(unit: &Unit) -> Vec<(&'static str, Vec<u8>)> {
    let mut frames = vec![("format", FORMAT.as_bytes().to_vec())];
    match unit {
        Unit::User(id) => {
            frames.push(("unit", b"user".to_vec()));
            frames.push(("user_id", id.as_bytes().to_vec()));
        }
        Unit::Shared => frames.push(("unit", b"shared".to_vec())),
    }
    frames
}

/// Walk `unit` on `backend` into `out`, in canonical frame order. Opens read-only;
/// an absent store contributes nothing. Decode failures are counted, not skipped —
/// the caller decides what a non-zero count means.
pub fn extract<W: Write>(
    base: &Path,
    backend: StorageBackend,
    unit: &Unit,
    out: W,
) -> Result<Extraction> {
    extract_with(base, backend, unit, out, None, Sealed::AsStored)
}

fn extract_with<W: Write>(
    base: &Path,
    backend: StorageBackend,
    unit: &Unit,
    out: W,
    quarantine: Option<&mut QuarantineWriter>,
    sealed: Sealed<'_>,
) -> Result<Extraction> {
    run_sink(unit, out, quarantine, sealed, |sink| {
        match (backend, unit) {
            (StorageBackend::RocksDb, Unit::User(id)) => {
                let dir = user_dir(base, id)?;
                rocks_primary(sink, &dir.join("storage"))?;
                rocks_families(sink, &dir.join("graph/graph"), |cf| format!("graph/{cf}"))
            }
            (StorageBackend::RocksDb, Unit::Shared) => {
                rocks_families(sink, &base.join("shared"), |cf| {
                    if cf == "audit" {
                        "audit/audit".to_string()
                    } else {
                        format!("shared/{cf}")
                    }
                })
            }
            #[cfg(feature = "storage-redb")]
            (StorageBackend::Redb, _) => redb_unit(sink, base, unit, &LiveRedb::default()),
            #[cfg(not(feature = "storage-redb"))]
            (StorageBackend::Redb, _) => {
                bail!("this build has no redb backend (feature storage-redb)")
            }
        }
    })
}

fn run_sink<'q, W: Write>(
    unit: &Unit,
    out: W,
    quarantine: Option<&'q mut QuarantineWriter>,
    sealed: Sealed<'q>,
    body: impl FnOnce(&mut Sink<'q, W>) -> Result<()>,
) -> Result<Extraction> {
    let mut sink = Sink {
        writer: PortableWriter::new(out)?,
        stats: Extraction::default(),
        quarantine,
        sealed,
    };
    for (key, value) in meta_frames(unit) {
        sink.frame(META, key.as_bytes(), &value)?;
    }
    body(&mut sink)?;
    let mut stats = std::mem::take(&mut sink.stats);
    let (frames, digest) = sink.writer.finish()?;
    stats.frames = frames;
    stats.digest = portable::hex(&digest);
    Ok(stats)
}

/// Handles this process holds open (redb admits one opener per file); `None` reads the file read-only from disk.
#[cfg(feature = "storage-redb")]
#[derive(Clone, Copy, Default)]
pub struct LiveRedb<'a> {
    pub primary: Option<&'a redb::Database>,
    pub graph: Option<&'a redb::Database>,
    pub audit: Option<&'a redb::Database>,
    pub shared: Option<&'a redb::Database>,
}

#[cfg(feature = "storage-redb")]
fn redb_unit<W: Write>(
    sink: &mut Sink<W>,
    base: &Path,
    unit: &Unit,
    live: &LiveRedb,
) -> Result<()> {
    match unit {
        Unit::User(id) => {
            let dir = user_dir(base, id)?;
            let primary = dir.join("storage.redb");
            match live.primary {
                Some(db) => redb_primary_db(sink, db, &primary)?,
                None => redb_primary(sink, &primary)?,
            }
            let graph = |t: &str| Some(format!("graph/{t}"));
            let graph_path = dir.join("graph/graph.redb");
            match live.graph {
                Some(db) => redb_tables_raw_db(sink, db, &graph_path, graph),
                None => redb_tables_raw(sink, &graph_path, graph),
            }
        }
        Unit::Shared => {
            let audit = |t: &str| (t == "audit").then(|| "audit/audit".to_string());
            let audit_path = base.join("audit.redb");
            match live.audit {
                Some(db) => redb_tables_raw_db(sink, db, &audit_path, audit)?,
                None => redb_tables_raw(sink, &audit_path, audit)?,
            }
            let shared = |t: &str| (t != "audit").then(|| format!("shared/{t}"));
            let shared_path = base.join("shared.redb");
            match live.shared {
                Some(db) => redb_tables_raw_db(sink, db, &shared_path, shared),
                None => redb_tables_raw(sink, &shared_path, shared),
            }
        }
    }
}

/// Same frames, order and digest as an offline `extract` of the same rows.
#[cfg(feature = "storage-redb")]
pub fn extract_live<W: Write>(
    base: &Path,
    unit: &Unit,
    live: &LiveRedb,
    out: W,
) -> Result<Extraction> {
    run_sink(unit, out, None, Sealed::AsStored, |sink| {
        redb_unit(sink, base, unit, live)
    })
}

#[cfg(feature = "rocksdb")]
fn refuse_if_encrypted(storage_dir: &Path) -> Result<()> {
    let keystore = storage_dir.join(KEYSTORE);
    if exists(&keystore)? {
        bail!(
            "{} uses record-level encryption ({} is present). An export or import does not \
             carry a keystore; `veld migrate` moves an encrypted store between engines — refusing.",
            storage_dir.display(),
            keystore.display()
        );
    }
    Ok(())
}

#[cfg(not(feature = "rocksdb"))]
const NO_ROCKSDB: &str =
    "this build has no RocksDB support; rebuild veld with `--features rocksdb` to read or write a RocksDB store";

#[cfg(not(feature = "rocksdb"))]
fn rocks_primary<W: Write>(_sink: &mut Sink<W>, _dir: &Path) -> Result<()> {
    bail!(NO_ROCKSDB)
}

#[cfg(not(feature = "rocksdb"))]
fn rocks_families<W: Write>(
    _sink: &mut Sink<W>,
    _dir: &Path,
    _logical: impl Fn(&str) -> String,
) -> Result<()> {
    bail!(NO_ROCKSDB)
}

#[cfg(not(feature = "rocksdb"))]
fn apply_rocks_raw(
    _dir: &Path,
    _records: Vec<PortableRecord>,
    _family_for: impl Fn(&str) -> Option<String>,
) -> Result<()> {
    bail!(NO_ROCKSDB)
}

#[cfg(not(feature = "rocksdb"))]
fn apply_rocks_primary(
    _user_dir: &Path,
    _records: Vec<PortableRecord>,
    _sealed: bool,
) -> Result<()> {
    bail!(NO_ROCKSDB)
}

#[cfg(feature = "rocksdb")]
fn rocks_primary<W: Write>(sink: &mut Sink<W>, dir: &Path) -> Result<()> {
    use rocksdb::{Direction, IteratorMode};
    if !sink.opens_sealed() {
        refuse_if_encrypted(dir)?;
    }
    let Some(store) = readonly::open_rocks(dir)? else {
        return Ok(());
    };
    if let Some(cf) = store
        .column_families
        .iter()
        .find(|cf| !["default", "kv", "memory_index"].contains(&cf.as_str()))
    {
        bail!("{} has column family '{cf}', which this migration does not know how to carry — refusing rather than dropping it", dir.display());
    }
    let db = &store.db;
    let default = db
        .cf_handle("default")
        .ok_or_else(|| anyhow!("{} has no default column family", dir.display()))?;
    let is_memory = crate::memory::storage::is_memory_key;
    let index = db.cf_handle("memory_index");
    let carries_sentinel = sink.keys().is_some();
    let mut sentinel = match sink.keys() {
        None => None,
        Some(keys) => {
            if db.get_cf(default, GENERATION_SENTINEL)?.is_some() {
                bail!("{} holds a keystore generation sentinel in its default family, where rocksdb never reads one — refusing to guess which is current", dir.display());
            }
            let stored = match index {
                Some(cf) => db.get_cf(cf, GENERATION_SENTINEL)?,
                None => None,
            };
            sentinel_frame(keys, stored, dir)?
        }
    };

    for item in db.iterator_cf(default, IteratorMode::Start) {
        let (key, value) = item?;
        if is_memory(&key) || key.starts_with(VMAPPING) {
            continue;
        }
        sink.put_default(&key, &value, &mut sentinel)?;
    }
    sink.flush_sentinel(&mut sentinel)?;
    if let Some(kv) = db.cf_handle("kv") {
        for item in db.iterator_cf(kv, IteratorMode::Start) {
            let (key, value) = item?;
            sink.put("primary/kv", &key, &value)?;
        }
    }
    for item in db.iterator_cf(default, IteratorMode::Start) {
        let (key, value) = item?;
        if !is_memory(&key) {
            continue;
        }
        let label = || {
            format!(
                "primary/memories {}",
                uuid::Uuid::from_slice(&key)
                    .map(|u| u.to_string())
                    .unwrap_or_default()
            )
        };
        if sink.opens_sealed() && crate::keystore::is_encrypted_record(&value) {
            sink.sealed_memory(&key, &value, label())?;
            continue;
        }
        match crate::memory::storage::portable_memory_value(&value) {
            Ok(bytes) => sink.put("primary/memories", &key, &bytes)?,
            Err(_) => sink.fail("primary/memories", &key, &value, label())?,
        }
    }
    let mut mappings = Vec::new();
    for item in db.iterator_cf(default, IteratorMode::From(VMAPPING, Direction::Forward)) {
        let (key, value) = item?;
        if !key.starts_with(VMAPPING) {
            break;
        }
        let id = std::str::from_utf8(&key[VMAPPING.len()..])
            .ok()
            .and_then(|s| uuid::Uuid::parse_str(s).ok().filter(|u| u.to_string() == s));
        match (
            id,
            crate::memory::storage::check_vector_mapping_value(&value),
        ) {
            (Some(id), Ok(())) => mappings.push((id.into_bytes(), value.to_vec())),
            _ => sink.fail(
                "primary/vector_mappings",
                &key,
                &value,
                format!("primary/vector_mappings {}", String::from_utf8_lossy(&key)),
            )?,
        }
    }
    mappings.sort();
    for (id, value) in mappings {
        sink.put("primary/vector_mappings", &id, &value)?;
    }
    if let Some(index) = index {
        for item in db.iterator_cf(index, IteratorMode::Start) {
            let (key, _) = item?;
            if carries_sentinel && &*key == GENERATION_SENTINEL {
                continue;
            }
            sink.stats.derived_rows_not_carried += 1;
        }
    }
    Ok(())
}

#[cfg(feature = "rocksdb")]
fn rocks_families<W: Write>(
    sink: &mut Sink<W>,
    dir: &Path,
    logical: impl Fn(&str) -> String,
) -> Result<()> {
    let Some(store) = readonly::open_rocks(dir)? else {
        return Ok(());
    };
    let mut families: Vec<(String, &String)> = store
        .column_families
        .iter()
        .map(|cf| (logical(cf), cf))
        .collect();
    families.sort();
    for (name, cf) in families {
        let handle = store
            .db
            .cf_handle(cf)
            .ok_or_else(|| anyhow!("{} lost column family '{cf}' while open", dir.display()))?;
        for item in store.db.iterator_cf(handle, rocksdb::IteratorMode::Start) {
            let (key, value) = item?;
            sink.put(&name, &key, &value)?;
        }
    }
    Ok(())
}

#[cfg(feature = "storage-redb")]
type RawTable<'a> = redb::TableDefinition<'a, &'static [u8], &'static [u8]>;

/// The table an import is writing, by name, so consecutive records for one table share
/// its handle.
#[cfg(feature = "storage-redb")]
type OpenRawTable<'name, 'txn> = (&'name str, redb::Table<'txn, &'static [u8], &'static [u8]>);

#[cfg(feature = "storage-redb")]
fn redb_table_names(read: &redb::ReadTransaction, path: &Path) -> Result<Vec<String>> {
    use redb::{MultimapTableHandle, TableHandle};
    if let Some(t) = read.list_multimap_tables()?.next() {
        bail!("{} has multimap table '{}', which this migration does not carry — refusing rather than dropping it", path.display(), t.name());
    }
    let mut names: Vec<String> = read.list_tables()?.map(|t| t.name().to_string()).collect();
    names.sort();
    Ok(names)
}

#[cfg(feature = "storage-redb")]
fn redb_primary<W: Write>(sink: &mut Sink<W>, path: &Path) -> Result<()> {
    let Some(store) = readonly::open_redb(path)? else {
        return Ok(());
    };
    redb_primary_db(sink, &store.db, path)
}

#[cfg(feature = "storage-redb")]
fn redb_primary_db<W: Write>(sink: &mut Sink<W>, db: &redb::Database, path: &Path) -> Result<()> {
    use redb::ReadableTable;
    use redb::ReadableTableMetadata;
    let read = db.begin_read()?;
    let names = redb_table_names(&read, path)?;
    let derived = crate::storage::redb::DERIVED_INDEX_TABLES;
    if let Some(t) = names.iter().find(|t| {
        !["default", "kv", "memories", "vector_mappings"].contains(&t.as_str())
            && !derived.contains(&t.as_str())
    }) {
        bail!("{} has table '{t}', which this migration does not know how to carry — refusing rather than dropping it", path.display());
    }
    for table in names.iter().filter(|t| derived.contains(&t.as_str())) {
        sink.stats.derived_rows_not_carried += read.open_table(RawTable::new(table))?.len()?;
    }
    let has = |table: &str| names.iter().any(|n| n == table);
    let mut sentinel = match sink.keys() {
        Some(keys) if keys.generation.is_some() => {
            let stored = if has("default") {
                read.open_table(RawTable::new("default"))?
                    .get(GENERATION_SENTINEL)?
                    .map(|v| v.value().to_vec())
            } else {
                None
            };
            sentinel_frame(keys, stored, path)?
        }
        _ => None,
    };
    for table in ["default", "kv", "memories", "vector_mappings"] {
        if table != "default" {
            sink.flush_sentinel(&mut sentinel)?;
        }
        if !has(table) {
            continue;
        }
        let handle = read.open_table(RawTable::new(table))?;
        for row in handle.iter()? {
            let (k, v) = row?;
            let (key, value) = (k.value(), v.value());
            match table {
                "default" => sink.put_default(key, value, &mut sentinel)?,
                "memories"
                    if sink.opens_sealed() && crate::keystore::is_encrypted_record(value) =>
                {
                    sink.sealed_memory(key, value, format!("primary/memories {}", redb_label(key)))?
                }
                // A backup restores into the same store, under the same keystore.
                "memories" if key.len() == 16 && crate::keystore::is_encrypted_record(value) => {
                    sink.put("primary/memories", key, value)?
                }
                "memories" => match crate::memory::storage::portable_memory_value(value) {
                    Ok(bytes) if key.len() == 16 => sink.put("primary/memories", key, &bytes)?,
                    _ => sink.fail(
                        "primary/memories",
                        key,
                        value,
                        format!("primary/memories {}", redb_label(key)),
                    )?,
                },
                "vector_mappings" => {
                    match crate::memory::storage::check_vector_mapping_value(value) {
                        Ok(()) if key.len() == 16 => {
                            sink.put("primary/vector_mappings", key, value)?
                        }
                        _ => sink.fail(
                            "primary/vector_mappings",
                            key,
                            value,
                            format!("primary/vector_mappings {}", redb_label(key)),
                        )?,
                    }
                }
                other => sink.put(&format!("primary/{other}"), key, value)?,
            }
        }
    }
    Ok(())
}

/// `logical` maps a table to its keyspace; `None` means the table does not belong in
/// this unit, which is refused if it holds rows and ignored if it is empty.
#[cfg(feature = "storage-redb")]
fn redb_tables_raw<W: Write>(
    sink: &mut Sink<W>,
    path: &Path,
    logical: impl Fn(&str) -> Option<String>,
) -> Result<()> {
    let Some(store) = readonly::open_redb(path)? else {
        return Ok(());
    };
    redb_tables_raw_db(sink, &store.db, path, logical)
}

#[cfg(feature = "storage-redb")]
fn redb_tables_raw_db<W: Write>(
    sink: &mut Sink<W>,
    db: &redb::Database,
    path: &Path,
    logical: impl Fn(&str) -> Option<String>,
) -> Result<()> {
    use redb::{ReadableTable, ReadableTableMetadata};
    let read = db.begin_read()?;
    let mut tables = Vec::new();
    for name in redb_table_names(&read, path)? {
        match logical(&name) {
            Some(keyspace) => tables.push((keyspace, name)),
            None => {
                if read.open_table(RawTable::new(&name))?.len()? > 0 {
                    bail!("{} has rows in table '{name}', which does not belong there — refusing rather than dropping them", path.display());
                }
            }
        }
    }
    tables.sort();
    for (keyspace, name) in tables {
        let handle = read.open_table(RawTable::new(&name))?;
        for row in handle.iter()? {
            let (k, v) = row?;
            sink.put(&keyspace, k.value(), v.value())?;
        }
    }
    Ok(())
}

#[cfg(feature = "storage-redb")]
fn redb_label(key: &[u8]) -> String {
    uuid::Uuid::from_slice(key)
        .map(|u| u.to_string())
        .unwrap_or_else(|_| key.iter().map(|b| format!("{b:02x}")).collect())
}

// ─── export ─────────────────────────────────────────────────────────────────────

pub(crate) fn dump_file_name(unit: &Unit) -> String {
    match unit {
        Unit::User(id) => format!("users/{id}.veldport"),
        Unit::Shared => "shared.veldport".to_string(),
    }
}

/// A unit's undecodable records, set aside rather than exported.
#[derive(Debug, Clone, Serialize)]
pub struct QuarantineReport {
    pub unit: Unit,
    pub records: u64,
    pub ids: Vec<String>,
    /// VELDPORT file holding each record's raw key and value, byte for byte.
    pub dump: PathBuf,
    /// JSON listing: keyspace, key, value length and value sha256 per record.
    pub listing: PathBuf,
}

fn quarantine_paths(dir: &Path, unit: &Unit) -> (PathBuf, PathBuf) {
    let stem = dump_file_name(unit);
    let stem = stem.trim_end_matches(".veldport");
    (
        dir.join(format!("{stem}.quarantine.veldport")),
        dir.join(format!("{stem}.quarantine.json")),
    )
}

/// The directory must be absent or empty, and is created owner-only: it holds raw records.
fn prepare_private_dir(dir: &Path, what: &str) -> Result<()> {
    match std::fs::read_dir(dir) {
        Ok(mut entries) => {
            if entries.next().is_some() {
                bail!("{what} {} is not empty", dir.display());
            }
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let mut builder = std::fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
            builder
                .create(dir)
                .with_context(|| format!("creating {what} {}", dir.display()))
        }
        Err(e) => Err(e).with_context(|| format!("reading {what} {}", dir.display())),
    }
}

fn export_unit(
    base: &Path,
    backend: StorageBackend,
    unit: &Unit,
    path: &Path,
    quarantine_dir: Option<&Path>,
    sealed: Sealed<'_>,
) -> Result<(Extraction, Option<QuarantineReport>)> {
    write_dump(unit, path, quarantine_dir, |out, quarantine| {
        extract_with(base, backend, unit, out, quarantine, sealed)
    })
}

fn write_dump(
    unit: &Unit,
    path: &Path,
    quarantine_dir: Option<&Path>,
    extract_into: impl FnOnce(
        &mut std::io::BufWriter<std::fs::File>,
        Option<&mut QuarantineWriter>,
    ) -> Result<Extraction>,
) -> Result<(Extraction, Option<QuarantineReport>)> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let partial = path.with_extension("partial");
    let q_paths = quarantine_dir.map(|d| quarantine_paths(d, unit));
    let result = (|| -> Result<(Extraction, Option<QuarantineReport>)> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .with_context(|| format!("creating {}", partial.display()))?;
        let mut out = std::io::BufWriter::new(file);
        let mut quarantine = match &q_paths {
            Some((dump, _)) => {
                if let Some(parent) = dump.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let f = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(dump.with_extension("partial"))
                    .with_context(|| format!("creating {}", dump.display()))?;
                Some(QuarantineWriter {
                    writer: PortableWriter::new(std::io::BufWriter::new(f))?,
                    listing: Vec::new(),
                })
            }
            None => None,
        };
        let stats = extract_into(&mut out, quarantine.as_mut())?;
        out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        if stats.decode_failures > 0 {
            bail!(
                "{unit}: {} record(s) failed to decode and were not exported — the export is refused, not shortened \
                 (pass --quarantine-undecodable <dir> to set them aside explicitly). Undecodable: {}",
                stats.decode_failures,
                stats.failed_keys.join(", ")
            );
        }
        let quarantine_report = match (quarantine, &q_paths) {
            (Some(q), Some((dump, listing))) if stats.quarantined > 0 => {
                let (count, digest) = q.writer.finish()?;
                if count != stats.quarantined {
                    bail!(
                        "{unit}: quarantine wrote {count} records but {} were set aside",
                        stats.quarantined
                    );
                }
                std::fs::rename(dump.with_extension("partial"), dump)?;
                let body = serde_json::json!({
                    "unit": unit,
                    "records": count,
                    "dump": dump.file_name().map(|n| n.to_string_lossy().to_string()),
                    "dump_digest": portable::hex(&digest),
                    "entries": q.listing,
                });
                write_atomic(listing, &serde_json::to_vec_pretty(&body)?)?;
                Some(QuarantineReport {
                    unit: unit.clone(),
                    records: count,
                    ids: stats.quarantined_keys.clone(),
                    dump: dump.clone(),
                    listing: listing.clone(),
                })
            }
            (_, Some((dump, _))) => {
                std::fs::remove_file(dump.with_extension("partial"))?;
                None
            }
            _ => None,
        };
        std::fs::rename(&partial, path)
            .with_context(|| format!("renaming {} into place", partial.display()))?;
        let (frames, digest) = portable::read_trailer(path)?;
        if frames != stats.frames || portable::hex(&digest) != stats.digest {
            bail!(
                "{}: the trailer on disk disagrees with what was written",
                path.display()
            );
        }
        Ok((stats, quarantine_report))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&partial);
        let _ = std::fs::remove_file(path);
        if let Some((dump, listing)) = &q_paths {
            let _ = std::fs::remove_file(dump.with_extension("partial"));
            let _ = std::fs::remove_file(dump);
            let _ = std::fs::remove_file(listing);
        }
    }
    result
}

/// Write one dump per unit under `out_dir`, plus a manifest. `out_dir` must be absent
/// or empty. With `quarantine_dir`, undecodable records are written byte for byte there
/// instead of failing the export; without it they fail it. Any failure removes every
/// file this call wrote.
pub fn export(
    base: &Path,
    backend: StorageBackend,
    scope: &Scope,
    out_dir: &Path,
    quarantine_dir: Option<&Path>,
) -> Result<Vec<DumpReport>> {
    match std::fs::read_dir(out_dir) {
        Ok(mut entries) => {
            if entries.next().is_some() {
                bail!(
                    "{} is not empty — an export directory holds exactly one export",
                    out_dir.display()
                );
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => std::fs::create_dir_all(out_dir)?,
        Err(e) => return Err(e).with_context(|| format!("reading {}", out_dir.display())),
    }
    if let Some(q) = quarantine_dir {
        prepare_private_dir(q, "quarantine directory")?;
    }
    let units = resolve_units(base, backend, scope)?;
    let mut reports = Vec::new();
    let mut failures = Vec::new();
    for unit in &units {
        let path = out_dir.join(dump_file_name(unit));
        match export_unit(base, backend, unit, &path, quarantine_dir, Sealed::AsStored) {
            Ok((stats, quarantine)) => reports.push(DumpReport {
                unit: unit.clone(),
                path,
                frames: stats.frames,
                records: stats.records,
                digest: stats.digest.clone(),
                derived_rows_not_carried: stats.derived_rows_not_carried,
                quarantined: stats.quarantined,
                source_records: stats.source_records(),
                quarantine,
            }),
            Err(e) => failures.push(format!("{e:#}")),
        }
    }
    let manifest_result = if failures.is_empty() {
        write_export_manifest(out_dir, backend, &reports)
    } else {
        Ok(())
    };
    if let Err(e) = manifest_result {
        failures.push(format!("{e:#}"));
    }
    if !failures.is_empty() {
        for r in &reports {
            let _ = std::fs::remove_file(&r.path);
            if let Some(q) = &r.quarantine {
                let _ = std::fs::remove_file(&q.dump);
                let _ = std::fs::remove_file(&q.listing);
            }
        }
        let _ = std::fs::remove_dir(out_dir.join("users"));
        if let Some(q) = quarantine_dir {
            let _ = std::fs::remove_dir(q.join("users"));
        }
        bail!(
            "export failed; no dump was left behind:\n  {}",
            failures.join("\n  ")
        );
    }
    Ok(reports)
}

fn write_export_manifest(
    out_dir: &Path,
    backend: StorageBackend,
    reports: &[DumpReport],
) -> Result<()> {
    let manifest = Manifest {
        format: FORMAT.to_string(),
        source_backend: backend.to_string(),
        dumps: reports
            .iter()
            .map(|r| ManifestEntry {
                file: dump_file_name(&r.unit),
                unit: r.unit.clone(),
                frames: r.frames,
                digest: r.digest.clone(),
            })
            .collect(),
    };
    write_atomic(
        &out_dir.join(MANIFEST),
        &serde_json::to_vec_pretty(&manifest)?,
    )
}

/// No quarantine here: an undecodable record refuses the export rather than shortening a backup.
#[cfg(feature = "storage-redb")]
pub fn export_live(
    base: &Path,
    units: &[Unit],
    live: &LiveRedb,
    out_dir: &Path,
) -> Result<Vec<DumpReport>> {
    prepare_private_dir(out_dir, "export directory")?;
    let mut reports = Vec::new();
    let result = (|| -> Result<()> {
        for unit in units {
            let path = out_dir.join(dump_file_name(unit));
            let (stats, _) = write_dump(unit, &path, None, |out, _| {
                extract_live(base, unit, live, out)
            })?;
            reports.push(DumpReport {
                unit: unit.clone(),
                path,
                frames: stats.frames,
                records: stats.records,
                digest: stats.digest.clone(),
                derived_rows_not_carried: stats.derived_rows_not_carried,
                quarantined: 0,
                source_records: stats.source_records(),
                quarantine: None,
            });
        }
        write_export_manifest(out_dir, StorageBackend::Redb, &reports)
    })();
    if let Err(e) = result {
        for r in &reports {
            let _ = std::fs::remove_file(&r.path);
        }
        let _ = std::fs::remove_dir(out_dir.join("users"));
        return Err(e);
    }
    Ok(reports)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("partial");
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// ─── import ─────────────────────────────────────────────────────────────────────

fn parse_dump(records: Vec<PortableRecord>) -> Result<(Unit, Vec<PortableRecord>)> {
    let mut iter = records.into_iter().peekable();
    let mut meta = BTreeMap::new();
    while let Some(rec) = iter.next_if(|r| r.cf == META) {
        let key = String::from_utf8(rec.key).context("meta key is not utf-8")?;
        let value = String::from_utf8(rec.value).context("meta value is not utf-8")?;
        if meta.insert(key.clone(), value).is_some() {
            bail!("dump repeats meta key '{key}'");
        }
    }
    match meta.get("format").map(String::as_str) {
        Some(FORMAT) => {}
        Some(other) => bail!("dump format is '{other}', this build reads '{FORMAT}'"),
        None => bail!("dump has no format frame — not a migration dump (a backup dump is a physical copy of one engine and does not migrate)"),
    }
    let unit = match (meta.get("unit").map(String::as_str), meta.get("user_id")) {
        (Some("user"), Some(id)) => {
            crate::path_guard::sanitize_component(id, "user id")?;
            Unit::User(id.clone())
        }
        (Some("shared"), None) => Unit::Shared,
        _ => bail!("dump meta does not name a unit"),
    };
    let data: Vec<PortableRecord> = iter.collect();
    let allowed: &[&str] = match unit {
        Unit::User(_) => &["primary/", "graph/"],
        Unit::Shared => &["audit/", "shared/"],
    };
    if let Some(rec) = data.iter().find(|r| {
        !allowed
            .iter()
            .any(|p| r.cf.starts_with(p) && r.cf.len() > p.len())
            || r.cf == META
    }) {
        bail!(
            "dump for {unit} carries keyspace '{}', which does not belong to it",
            rec.cf
        );
    }
    Ok((unit, data))
}

struct PendingImport {
    path: PathBuf,
    unit: Unit,
    frames: u64,
    digest: [u8; 32],
}

fn collect_dumps(input: &Path) -> Result<Vec<PathBuf>> {
    if !std::fs::metadata(input)
        .with_context(|| format!("opening {}", input.display()))?
        .is_dir()
    {
        return Ok(vec![input.to_path_buf()]);
    }
    let manifest_path = input.join(MANIFEST);
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(&manifest_path).with_context(|| {
            format!(
                "{} has no readable {MANIFEST} — not a complete export",
                input.display()
            )
        })?)
        .with_context(|| format!("parsing {}", manifest_path.display()))?;
    if manifest.format != FORMAT {
        bail!(
            "{} declares format '{}', this build reads '{FORMAT}'",
            manifest_path.display(),
            manifest.format
        );
    }
    let mut listed = Vec::new();
    for entry in &manifest.dumps {
        if entry.file != dump_file_name(&entry.unit) {
            bail!(
                "{MANIFEST} lists '{}' for {}, expected '{}'",
                entry.file,
                entry.unit,
                dump_file_name(&entry.unit)
            );
        }
        let path = input.join(&entry.file);
        let (frames, digest) = portable::read_trailer(&path)?;
        if frames != entry.frames || portable::hex(&digest) != entry.digest {
            bail!("{} does not match its {MANIFEST} entry", path.display());
        }
        listed.push(path);
    }
    let mut on_disk = Vec::new();
    for dir in [input.to_path_buf(), input.join("users")] {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
        };
        for entry in entries {
            let path = entry?.path();
            match path.extension().and_then(|e| e.to_str()) {
                Some("veldport") | Some("partial") => on_disk.push(path),
                _ => {}
            }
        }
    }
    on_disk.sort();
    let mut expected = listed.clone();
    expected.sort();
    if on_disk != expected {
        bail!("{} holds dump files its {MANIFEST} does not list (or an interrupted .partial) — refusing an ambiguous import", input.display());
    }
    Ok(listed)
}

/// Import every dump at `input` (a dump file, or an export directory with a manifest)
/// into `backend` under `base`, verifying each against its trailer. Every target must
/// be empty; that is checked for all of them before anything is written.
pub fn import(base: &Path, backend: StorageBackend, input: &Path) -> Result<Vec<ImportReport>> {
    let mut pending = Vec::new();
    for path in collect_dumps(input)? {
        let (frames, digest) = portable::read_trailer(&path)?;
        let records = portable::read_all(
            std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?,
        )
        .with_context(|| format!("reading {}", path.display()))?;
        let (unit, _) = parse_dump(records)?;
        if pending.iter().any(|p: &PendingImport| p.unit == unit) {
            bail!("two dumps name {unit}");
        }
        refuse_populated_target(base, backend, &unit)?;
        pending.push(PendingImport {
            path,
            unit,
            frames,
            digest,
        });
    }
    let mut reports = Vec::new();
    for p in pending {
        reports.push(import_one(base, backend, &p, None).with_context(|| {
            let done: Vec<String> = reports
                .iter()
                .map(|r: &ImportReport| r.unit.to_string())
                .collect();
            format!(
                "importing {} from {} (already imported and verified: [{}])",
                p.unit,
                p.path.display(),
                done.join(", ")
            )
        })?);
    }
    Ok(reports)
}

fn refuse_populated_target(base: &Path, backend: StorageBackend, unit: &Unit) -> Result<()> {
    let existing = extract(base, backend, unit, std::io::sink())
        .with_context(|| format!("inspecting the {backend} target for {unit}"))?;
    if existing.holds_anything() {
        bail!(
            "the {backend} target for {unit} is not empty ({} records, {} undecodable, {} index rows) — \
             refusing to merge into it. Import into an empty target, verify, then switch the configured backend.",
            existing.records,
            existing.decode_failures,
            existing.derived_rows_not_carried
        );
    }
    Ok(())
}

/// `keystore`: where migrate installed an encrypted unit's keystore on the target.
fn import_one(
    base: &Path,
    backend: StorageBackend,
    p: &PendingImport,
    keystore: Option<&Path>,
) -> Result<ImportReport> {
    let records = portable::read_all(std::fs::File::open(&p.path)?)?;
    let (unit, data) = parse_dump(records)?;
    if unit != p.unit {
        bail!(
            "{} changed between the preflight and the import",
            p.path.display()
        );
    }
    let record_count = data.len() as u64;
    apply(base, backend, &unit, data, keystore.is_some())?;
    let verified = match keystore {
        None => verify_with(base, backend, &unit, p.frames, &p.digest, Sealed::AsStored)?,
        Some(path) => {
            let json = std::fs::read_to_string(path)
                .with_context(|| format!("reading the keystore copied to {}", path.display()))?;
            let keys = crate::keystore::unseal_record_keys(&json)
                .with_context(|| format!("unsealing the keystore copied to {}", path.display()))?;
            let target = Keys {
                cryptors: &keys.cryptors,
                generation: None,
            };
            verify_with(
                base,
                backend,
                &unit,
                p.frames,
                &p.digest,
                Sealed::Opened(Some(target)),
            )?
        }
    };
    Ok(ImportReport {
        unit,
        source: p.path.clone(),
        frames: p.frames,
        records: record_count,
        digest: portable::hex(&p.digest),
        verified_frames: verified.frames,
        verified_digest: verified.digest,
        verified_sealed_records: verified.sealed_records,
    })
}

/// Re-derive the target's frames and compare them with the dump's trailer. The only
/// path to `Ok` is equality of both the count and the digest.
pub fn verify(
    base: &Path,
    backend: StorageBackend,
    unit: &Unit,
    frames: u64,
    digest: &[u8; 32],
) -> Result<Extraction> {
    verify_with(base, backend, unit, frames, digest, Sealed::AsStored)
}

fn verify_with(
    base: &Path,
    backend: StorageBackend,
    unit: &Unit,
    frames: u64,
    digest: &[u8; 32],
    sealed: Sealed<'_>,
) -> Result<Extraction> {
    let derived = extract_with(base, backend, unit, std::io::sink(), None, sealed)?;
    let expected = portable::hex(digest);
    if derived.frames != frames || derived.digest != expected || derived.decode_failures > 0 {
        bail!(
            "verification FAILED for {unit} on {backend}: the dump declares {frames} frames with digest {expected}, \
             the target re-derives {} frames with digest {} ({} undecodable). The target is not a faithful copy — \
             do NOT switch the configured backend; remove the target's rows before retrying.",
            derived.frames,
            derived.digest,
            derived.decode_failures
        );
    }
    Ok(derived)
}

fn split_keyspace<'a>(cf: &'a str, component: &str) -> Option<&'a str> {
    cf.strip_prefix(component)
        .and_then(|rest| rest.strip_prefix('/'))
}

/// `sealed`: an encrypted unit whose keystore migrate has already installed on the target.
fn apply(
    base: &Path,
    backend: StorageBackend,
    unit: &Unit,
    data: Vec<PortableRecord>,
    sealed: bool,
) -> Result<()> {
    match (backend, unit) {
        (StorageBackend::RocksDb, Unit::User(id)) => {
            let dir = user_dir(base, id)?;
            let (primary, graph): (Vec<_>, Vec<_>) =
                data.into_iter().partition(|r| r.cf.starts_with("primary/"));
            if !primary.is_empty() {
                apply_rocks_primary(&dir, primary, sealed)?;
            }
            if !graph.is_empty() {
                apply_rocks_raw(&dir.join("graph/graph"), graph, |cf| {
                    split_keyspace(cf, "graph").map(str::to_string)
                })?;
            }
        }
        (StorageBackend::RocksDb, Unit::Shared) => {
            apply_rocks_raw(&base.join("shared"), data, |cf| {
                if cf == "audit/audit" {
                    Some("audit".to_string())
                } else {
                    split_keyspace(cf, "shared")
                        .filter(|t| *t != "audit")
                        .map(str::to_string)
                }
            })?;
        }
        #[cfg(feature = "storage-redb")]
        (StorageBackend::Redb, Unit::User(id)) => {
            let dir = user_dir(base, id)?;
            let (primary, graph): (Vec<_>, Vec<_>) =
                data.into_iter().partition(|r| r.cf.starts_with("primary/"));
            if !primary.is_empty() {
                apply_redb(&dir.join("storage.redb"), &primary, |cf| {
                    split_keyspace(cf, "primary")
                        .filter(|t| ["default", "kv", "memories", "vector_mappings"].contains(t))
                })?;
            }
            if !graph.is_empty() {
                apply_redb(&dir.join("graph/graph.redb"), &graph, |cf| {
                    split_keyspace(cf, "graph")
                })?;
            }
        }
        #[cfg(feature = "storage-redb")]
        (StorageBackend::Redb, Unit::Shared) => {
            let (audit, shared): (Vec<_>, Vec<_>) =
                data.into_iter().partition(|r| r.cf.starts_with("audit/"));
            if !audit.is_empty() {
                apply_redb(&base.join("audit.redb"), &audit, |cf| {
                    (cf == "audit/audit").then_some("audit")
                })?;
            }
            if !shared.is_empty() {
                apply_redb(&base.join("shared.redb"), &shared, |cf| {
                    split_keyspace(cf, "shared").filter(|t| *t != "audit")
                })?;
            }
        }
        #[cfg(not(feature = "storage-redb"))]
        (StorageBackend::Redb, _) => bail!("this build has no redb backend (feature storage-redb)"),
    }
    Ok(())
}

#[cfg(feature = "storage-redb")]
fn apply_redb<'a>(
    path: &Path,
    records: &'a [PortableRecord],
    table_for: impl Fn(&'a str) -> Option<&'a str>,
) -> Result<()> {
    use redb::{ReadableTableMetadata, TableHandle};
    let tuning = crate::config::StorageTuning::process();
    let db = crate::storage::redb::create_redb_database(path, &tuning)
        .with_context(|| format!("opening {}", path.display()))?;
    let txn = crate::storage::redb::begin_write(&db, tuning.durability)?;
    {
        let names: Vec<String> = txn.list_tables()?.map(|t| t.name().to_string()).collect();
        for name in names {
            if txn.open_table(RawTable::new(&name))?.len()? > 0 {
                bail!(
                    "{} gained rows in table '{name}' after the preflight — refusing to merge",
                    path.display()
                );
            }
        }
        let mut current: Option<OpenRawTable<'_, '_>> = None;
        for rec in records {
            let table = table_for(rec.cf.as_str())
                .ok_or_else(|| anyhow!("keyspace '{}' has no table on redb", rec.cf))?;
            if current.as_ref().map(|(n, _)| *n) != Some(table) {
                current = Some((table, txn.open_table(RawTable::new(table))?));
            }
            if let Some((_, handle)) = current.as_mut() {
                handle.insert(rec.key.as_slice(), rec.value.as_slice())?;
            }
        }
    }
    txn.commit()?;
    // A migration reports success once, so the copy holds after a power loss at either
    // class, as the RocksDB import's write-ahead-log sync makes it hold there.
    crate::storage::redb::make_durable(&db, tuning.durability)
}

#[cfg(feature = "rocksdb")]
fn open_rocks_for_import(dir: &Path, families: &[String]) -> Result<rocksdb::DB> {
    std::fs::create_dir_all(dir)?;
    let mut opts = rocksdb::Options::default();
    opts.create_if_missing(true);
    opts.create_missing_column_families(true);
    let mut names: Vec<String> = families.to_vec();
    names.push("default".to_string());
    if exists(&dir.join("CURRENT"))? {
        names.extend(rocksdb::DB::list_cf(&opts, dir)?);
    }
    names.sort();
    names.dedup();
    let db = rocksdb::DB::open_cf(&opts, dir, &names)
        .with_context(|| format!("opening {} for import", dir.display()))?;
    for name in &names {
        let cf = db
            .cf_handle(name)
            .ok_or_else(|| anyhow!("column family '{name}' did not open"))?;
        if db
            .iterator_cf(cf, rocksdb::IteratorMode::Start)
            .next()
            .is_some()
        {
            bail!(
                "{} gained rows in '{name}' after the preflight — refusing to merge",
                dir.display()
            );
        }
    }
    Ok(db)
}

#[cfg(feature = "rocksdb")]
fn write_rocks(
    db: &rocksdb::DB,
    rows: impl Iterator<Item = Result<(String, Vec<u8>, Vec<u8>)>>,
) -> Result<()> {
    let mut batch = rocksdb::WriteBatch::default();
    for row in rows {
        let (family, key, value) = row?;
        let cf = db
            .cf_handle(&family)
            .ok_or_else(|| anyhow!("column family '{family}' is not open"))?;
        batch.put_cf(cf, key, value);
        if batch.len() >= ROCKS_BATCH_OPS {
            db.write(std::mem::take(&mut batch))?;
        }
    }
    if !batch.is_empty() {
        db.write(batch)?;
    }
    db.flush_wal(true)?;
    Ok(())
}

#[cfg(feature = "rocksdb")]
fn apply_rocks_raw(
    dir: &Path,
    records: Vec<PortableRecord>,
    family_for: impl Fn(&str) -> Option<String>,
) -> Result<()> {
    let mut families = Vec::new();
    for rec in &records {
        let family = family_for(&rec.cf)
            .ok_or_else(|| anyhow!("keyspace '{}' has no column family on rocksdb", rec.cf))?;
        if !families.contains(&family) {
            families.push(family);
        }
    }
    let db = open_rocks_for_import(dir, &families)?;
    write_rocks(
        &db,
        records.into_iter().map(|r| {
            let family = family_for(&r.cf).expect("resolved above");
            Ok((family, r.key, r.value))
        }),
    )?;
    for family in &families {
        let cf = db.cf_handle(family).expect("opened above");
        db.flush_cf(cf)?;
    }
    Ok(())
}

#[cfg(feature = "rocksdb")]
fn apply_rocks_primary(user_dir: &Path, records: Vec<PortableRecord>, sealed: bool) -> Result<()> {
    let storage = user_dir.join("storage");
    if !sealed {
        refuse_if_encrypted(&storage)?;
    } else if !exists(&storage.join(KEYSTORE))? {
        bail!(
            "{} has no keystore to open the sealed records with — refusing to let the index rebuild create one",
            storage.display()
        );
    }
    {
        let db = open_rocks_for_import(&storage, &["kv".to_string(), "memory_index".to_string()])?;
        write_rocks(
            &db,
            records.into_iter().map(|r| {
                let key = r.key;
                match r.cf.as_str() {
                    "primary/default" if sealed && key == GENERATION_SENTINEL => {
                        Ok(("memory_index".to_string(), key, r.value))
                    }
                    "primary/default" => Ok(("default".to_string(), key, r.value)),
                    "primary/kv" => Ok(("kv".to_string(), key, r.value)),
                    "primary/memories" if key.len() == 16 => {
                        Ok(("default".to_string(), key, r.value))
                    }
                    "primary/vector_mappings" if key.len() == 16 => {
                        let id = uuid::Uuid::from_slice(&key)?;
                        Ok((
                            "default".to_string(),
                            format!("vmapping:{id}").into_bytes(),
                            r.value,
                        ))
                    }
                    other => Err(anyhow!(
                        "dump row in '{other}' has a key rocksdb cannot hold ({} bytes)",
                        key.len()
                    )),
                }
            }),
        )?;
        for family in ["default", "kv", "memory_index"] {
            db.flush_cf(db.cf_handle(family).expect("opened above"))?;
        }
    }
    let rebuilt =
        crate::memory::storage::MemoryStorage::open_without_creating_keystore(user_dir, None)
            .context("reopening the imported rocksdb store to rebuild its index")?
            .rebuild_index()
            .context("rebuilding memory_index on the imported rocksdb store")?;
    tracing::info!(records = rebuilt, dir = %storage.display(), "memory_index rebuilt after import");
    Ok(())
}

// ─── migrate ────────────────────────────────────────────────────────────────────

/// Copy `scope` from `from` to `to` under one data directory, verifying every unit.
/// The source is only ever opened read-only. With `dry_run` nothing is written and the
/// report says what a real run would find; without it, any blocked unit refuses the
/// whole migration before a single row is written.
///
/// An encrypted unit needs its keys, in the dry run too: the plan is only as good as
/// the proof that every sealed record opens under the keystore that goes with it.
pub fn migrate(
    base: &Path,
    from: StorageBackend,
    to: StorageBackend,
    scope: &Scope,
    dry_run: bool,
    quarantine_dir: Option<&Path>,
) -> Result<MigrateReport> {
    if from == to {
        bail!("--from and --to are both {from}; a migration needs two backends");
    }
    let units = resolve_units(base, from, scope)?;
    let mut report = MigrateReport {
        from: from.to_string(),
        to: to.to_string(),
        dry_run,
        quarantine_requested: quarantine_dir.is_some(),
        units: Vec::new(),
        imported: Vec::new(),
        quarantined: Vec::new(),
    };
    let mut keystores = Vec::new();
    for unit in units {
        let mut plan = UnitPlan {
            unit: unit.clone(),
            source: None,
            target: None,
            blocked: Vec::new(),
        };
        let keystore = match unit_keystore(base, from, to, &unit) {
            Ok(keystore) => Some(keystore),
            Err(e) => {
                plan.blocked.push(format!("{e:#}"));
                None
            }
        };
        if let Some(keystore) = &keystore {
            let sealed = keystore
                .as_ref()
                .map_or(Sealed::AsStored, |k| Sealed::Opened(Some(k.source_keys())));
            match extract_with(base, from, &unit, std::io::sink(), None, sealed) {
                Ok(source) => {
                    if source.decode_failures > 0 && quarantine_dir.is_none() {
                        plan.blocked.push(format!(
                            "{} source record(s) fail to decode: {}",
                            source.decode_failures,
                            source.failed_keys.join(", ")
                        ));
                    }
                    plan.source = Some(source);
                }
                Err(e) => plan.blocked.push(format!("source: {e:#}")),
            }
        }
        let target_sealed = match &keystore {
            Some(Some(_)) => Sealed::Opened(None),
            _ => Sealed::AsStored,
        };
        match extract_with(base, to, &unit, std::io::sink(), None, target_sealed) {
            Ok(target) => {
                if target.holds_anything() {
                    plan.blocked.push(format!(
                        "target is not empty ({} records, {} undecodable, {} index rows)",
                        target.records, target.decode_failures, target.derived_rows_not_carried
                    ));
                }
                plan.target = Some(target);
            }
            Err(e) => plan.blocked.push(format!("target: {e:#}")),
        }
        report.units.push(plan);
        keystores.push(keystore.flatten());
    }
    if dry_run {
        return Ok(report);
    }
    if report.blocked() {
        let reasons: Vec<String> = report
            .units
            .iter()
            .filter(|u| !u.blocked.is_empty())
            .map(|u| format!("{}: {}", u.unit, u.blocked.join("; ")))
            .collect();
        bail!(
            "migration refused before writing anything:\n  {}",
            reasons.join("\n  ")
        );
    }

    if let Some(q) = quarantine_dir {
        prepare_private_dir(q, "quarantine directory")?;
    }
    let work = WorkDir::create()?;
    for (plan, keystore) in report.units.iter().zip(&keystores) {
        let path = work.path.join(dump_file_name(&plan.unit));
        let sealed = keystore
            .as_ref()
            .map_or(Sealed::AsStored, |k| Sealed::Opened(Some(k.source_keys())));
        let (exported, quarantine) =
            export_unit(base, from, &plan.unit, &path, quarantine_dir, sealed)?;
        let planned = plan
            .source
            .as_ref()
            .map(Extraction::source_records)
            .unwrap_or(0);
        if exported.source_records() != planned {
            bail!(
                "{}: the source held {planned} records when planned and {} when exported — it changed during the migration",
                plan.unit,
                exported.source_records()
            );
        }
        report.quarantined.extend(quarantine);
        if let Some(keystore) = keystore {
            install_keystore(&keystore.target, &keystore.json)
                .with_context(|| format!("{}: installing the keystore", plan.unit))?;
        }
        let pending = PendingImport {
            path: path.clone(),
            unit: plan.unit.clone(),
            frames: exported.frames,
            digest: portable::read_trailer(&path)?.1,
        };
        let installed = keystore.as_ref().map(|k| k.target.as_path());
        let imported = import_one(base, to, &pending, installed).with_context(|| {
            let done: Vec<String> = report.imported.iter().map(|r| r.unit.to_string()).collect();
            format!(
                "migrating {} (already migrated and verified: [{}])",
                plan.unit,
                done.join(", ")
            )
        })?;
        let _ = std::fs::remove_file(&path);
        report.imported.push(ImportReport {
            source: PathBuf::new(),
            ..imported
        });
    }
    Ok(report)
}

fn keystore_path(user_dir: &Path, backend: StorageBackend) -> PathBuf {
    match backend {
        StorageBackend::RocksDb => user_dir.join("storage").join(KEYSTORE),
        StorageBackend::Redb => user_dir.join(KEYSTORE),
    }
}

/// An encrypted unit's keystore: the source file's text, where its copy goes, and the
/// keys it unseals to.
struct UnitKeystore {
    json: String,
    target: PathBuf,
    keys: crate::keystore::UnsealedRecordKeys,
}

impl UnitKeystore {
    fn source_keys(&self) -> Keys<'_> {
        Keys {
            cryptors: &self.keys.cryptors,
            generation: Some(self.keys.generation),
        }
    }
}

fn read_optional(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

fn same_keystore(a: &str, b: &str) -> bool {
    match (
        serde_json::from_str::<serde_json::Value>(a),
        serde_json::from_str::<serde_json::Value>(b),
    ) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// `Ok(None)` for a unit with no keystore. An encrypted unit whose keys are not at
/// hand, or whose target already holds another keystore, is an error.
fn unit_keystore(
    base: &Path,
    from: StorageBackend,
    to: StorageBackend,
    unit: &Unit,
) -> Result<Option<UnitKeystore>> {
    let Unit::User(id) = unit else {
        return Ok(None);
    };
    let dir = user_dir(base, id)?;
    let source = keystore_path(&dir, from);
    let Some(json) = read_optional(&source)? else {
        return Ok(None);
    };
    let set = |name: &str| std::env::var(name).is_ok_and(|v| !v.trim().is_empty());
    if !set("VELD_MASTER_PASSPHRASE") && !set("VELD_KMS_WRAP_KEY") {
        bail!(
            "the {from} store uses record-level encryption ({} is present) and neither \
             VELD_MASTER_PASSPHRASE nor VELD_KMS_WRAP_KEY is set. migrate opens every sealed \
             record under the keystore to verify the copy; set the one that unseals it.",
            source.display()
        );
    }
    let keys = crate::keystore::unseal_record_keys(&json).with_context(|| {
        format!(
            "{} does not unseal with the VELD_MASTER_PASSPHRASE or VELD_KMS_WRAP_KEY set here",
            source.display()
        )
    })?;
    let target = keystore_path(&dir, to);
    if let Some(existing) = read_optional(&target)? {
        if !same_keystore(&existing, &json) {
            bail!(
                "the {to} target already has a keystore at {} that differs from the source's {} — \
                 refusing to put records sealed under one beside the other",
                target.display(),
                source.display()
            );
        }
    }
    Ok(Some(UnitKeystore { json, target, keys }))
}

/// Write the source keystore's text at `path`, owner-only, never over another file. An
/// identical keystore already there is kept.
fn install_keystore(path: &Path, json: &str) -> Result<()> {
    match read_optional(path)? {
        Some(existing) if same_keystore(&existing, json) => return Ok(()),
        Some(_) => bail!(
            "{} holds a different keystore than the one planned — refusing to replace it",
            path.display()
        ),
        None => {}
    }
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", path.display()))?;
    std::fs::create_dir_all(parent)?;
    let staged = path.with_extension("json.migrate");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let result = (|| -> Result<()> {
        let mut file = options
            .open(&staged)
            .with_context(|| format!("creating {}", staged.display()))?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        std::fs::hard_link(&staged, path)
            .with_context(|| format!("placing the keystore at {}", path.display()))?;
        #[cfg(unix)]
        std::fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&staged);
    result
}

/// A private scratch directory for migrate's intermediate dumps, removed on drop:
/// the dumps are plaintext copies of every record.
struct WorkDir {
    path: PathBuf,
}

impl WorkDir {
    fn create() -> Result<Self> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path =
            std::env::temp_dir().join(format!("veld-migrate-{}-{nanos}", std::process::id()));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder
            .create(&path)
            .with_context(|| format!("creating scratch directory {}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(all(test, feature = "rocksdb"))]
pub(crate) const UNDECODABLE_VALUE: &[u8] = b"\xff\xfe\xfd this was never a memory";

/// Writes a 16-byte-keyed value into a user's rocksdb primary store that no format decoder
/// accepts. The runtime reader's raw-UTF-8 fallback still fabricates a memory from it,
/// which is the case a migration must refuse.
#[cfg(all(test, feature = "rocksdb"))]
pub(crate) fn plant_undecodable_rocks_memory(user_dir: &Path) -> uuid::Uuid {
    let bad = uuid::Uuid::new_v4();
    let db = rocksdb::DB::open_cf(
        &rocksdb::Options::default(),
        user_dir.join("storage"),
        ["default", "kv", "memory_index"],
    )
    .expect("open rocksdb store to plant a record");
    db.put(bad.as_bytes(), UNDECODABLE_VALUE).expect("plant");
    db.flush().expect("flush");
    bad
}

#[cfg(all(test, feature = "rocksdb"))]
mod tests;
