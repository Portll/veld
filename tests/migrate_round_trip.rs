#![cfg(all(feature = "rocksdb", feature = "storage-redb"))]
//! `veld migrate` there and back for a tenant with record-level encryption, the dry run
//! on that tenant, the refusals that name their reason, and the boot guard around a
//! completed migration through the real manager.
//!
//! Own test binary, because `VELD_MASTER_PASSPHRASE` is process-global.

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tempfile::TempDir;
use uuid::Uuid;

use veld::config::{ServerConfig, StorageBackend};
use veld::handlers::MultiUserMemoryManager;
use veld::keystore::{KdfParams, Keystore};
use veld::memory::storage::{Modality, VectorMappingEntry};
use veld::memory::{Experience, ExperienceType, Memory, MemoryId};
use veld::storage::boot_guard::refuse_stranded_data;
use veld::storage::legacy_rocksdb::RocksDbPrimaryMemoryStore;
use veld::storage::migrate::{self, MigrateReport, Scope};
use veld::storage::redb::RedbPrimaryMemoryStore;
use veld::storage::PrimaryMemoryStore;

static ENV_LOCK: Mutex<()> = Mutex::new(());

const PASSPHRASE: &str = "round-trip-correct-horse-battery-staple-R7";
const USER: &str = "rosa";
const PLAIN: &str = "written-in-plaintext-before-the-keystore";
const SECRET_BEFORE: &str = "sealed-before-rotation-do-not-leak-R7Q";
const SECRET_AFTER: &str = "sealed-after-rotation-do-not-leak-R7W";
const SECRET_TAG: &str = "secret-tag-do-not-leak-R7T";
const PLAIN_COUNT: usize = 10;
const SEALED_PER_EPOCH: usize = 15;
const SEALED_COUNT: u64 = 2 * SEALED_PER_EPOCH as u64;

/// The variables these tests set, cleared on entry and on drop so a failed test does
/// not leak a passphrase into the next.
struct Env(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

impl Env {
    fn lock() -> Self {
        let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_env();
        Env(guard)
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        clear_env();
    }
}

fn clear_env() {
    for name in [
        "VELD_MASTER_PASSPHRASE",
        "VELD_KMS_WRAP_KEY",
        "VELD_ALLOW_KEYSTORE_ROLLBACK",
    ] {
        env::remove_var(name);
    }
}

fn opposite(backend: StorageBackend) -> StorageBackend {
    match backend {
        StorageBackend::RocksDb => StorageBackend::Redb,
        StorageBackend::Redb => StorageBackend::RocksDb,
    }
}

fn sample(content: &str, tags: &[&str]) -> Memory {
    let experience = Experience {
        experience_type: ExperienceType::Observation,
        content: content.to_string(),
        tags: tags.iter().map(|t| t.to_string()).collect(),
        ..Default::default()
    };
    Memory::new(
        MemoryId(Uuid::new_v4()),
        experience,
        0.5,
        None,
        None,
        None,
        None,
    )
}

fn batch(count: usize, marker: &str, tag: &str) -> Vec<Memory> {
    (0..count)
        .map(|i| sample(&format!("{marker} {i}"), &[tag]))
        .collect()
}

fn user_dir(base: &Path) -> PathBuf {
    base.join(USER)
}

fn keystore_path(base: &Path, backend: StorageBackend) -> PathBuf {
    match backend {
        StorageBackend::RocksDb => user_dir(base).join("storage/keystore.json"),
        StorageBackend::Redb => user_dir(base).join("keystore.json"),
    }
}

fn engine_paths(base: &Path, backend: StorageBackend) -> Vec<PathBuf> {
    let dir = user_dir(base);
    match backend {
        StorageBackend::RocksDb => vec![dir.join("storage"), dir.join("graph/graph")],
        StorageBackend::Redb => vec![
            dir.join("storage.redb"),
            dir.join("graph/graph.redb"),
            dir.join("keystore.json"),
        ],
    }
}

fn rotate(keystore: &Path) {
    let mut ks = Keystore::from_json(&std::fs::read_to_string(keystore).unwrap()).unwrap();
    let kek = ks.unseal_with_passphrase(PASSPHRASE).unwrap();
    assert_eq!(ks.rotate_dek(&kek).unwrap(), 1);
    ks.save_to_path(keystore).unwrap();
}

fn mapping() -> VectorMappingEntry {
    let mut entry = VectorMappingEntry::default();
    entry.add_modality(Modality::Text, vec![3, 4]);
    entry
}

fn write_memories(backend: StorageBackend, dir: &Path, memories: &[Memory]) {
    fn put(store: &impl PrimaryMemoryStore, memories: &[Memory]) {
        for m in memories {
            store.put_memory(m).unwrap();
            store.put_vector_mapping(&m.id, &mapping()).unwrap();
            store
                .key_value_store()
                .put_bytes(format!("note-{}", m.id.0).as_bytes(), b"annotation")
                .unwrap();
        }
    }
    match backend {
        StorageBackend::RocksDb => put(
            &RocksDbPrimaryMemoryStore::open(dir, None).unwrap(),
            memories,
        ),
        StorageBackend::Redb => put(&RedbPrimaryMemoryStore::open(dir).unwrap(), memories),
    }
}

/// An encryption-v2 tenant: plaintext rows written before the keystore existed, then
/// `SEALED_PER_EPOCH` sealed at epoch 0, a rotation, and `SEALED_PER_EPOCH` sealed at
/// epoch 1. `VELD_MASTER_PASSPHRASE` stays set. Returns every memory written.
fn encrypted_origin(base: &Path, backend: StorageBackend) -> Vec<Memory> {
    let dir = user_dir(base);
    std::fs::create_dir_all(&dir).unwrap();
    let plain = batch(PLAIN_COUNT, PLAIN, "plain");
    let before = batch(SEALED_PER_EPOCH, SECRET_BEFORE, SECRET_TAG);
    let after = batch(SEALED_PER_EPOCH, SECRET_AFTER, "after");

    write_memories(backend, &dir, &plain);
    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);
    write_memories(backend, &dir, &before);
    rotate(&keystore_path(base, backend));
    write_memories(backend, &dir, &after);
    [plain, before, after].concat()
}

fn contains(haystack: &[u8], needle: &str) -> bool {
    haystack
        .windows(needle.len())
        .any(|w| w == needle.as_bytes())
}

fn files_under(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries {
            out.extend(files_under(&entry.unwrap().path()));
        }
    }
    out
}

/// Every file at `paths` that holds any of the secrets in the clear.
fn leaks(paths: &[PathBuf]) -> Vec<PathBuf> {
    paths
        .iter()
        .flat_map(|p| files_under(p))
        .filter(|f| {
            let bytes = std::fs::read(f).unwrap();
            [SECRET_BEFORE, SECRET_AFTER, SECRET_TAG]
                .iter()
                .any(|s| contains(&bytes, s))
        })
        .collect()
}

/// Every file at `paths` with its bytes, so "unchanged" is measured, not assumed.
fn bytes_at(paths: &[PathBuf]) -> BTreeMap<PathBuf, Vec<u8>> {
    paths
        .iter()
        .flat_map(|p| files_under(p))
        .map(|f| {
            let bytes = std::fs::read(&f).unwrap();
            (f, bytes)
        })
        .collect()
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// A copy of `from` without the `source` engine, so the next hop starts from only what
/// the previous migration produced.
fn carry_over_target_only(from: &Path, to: &Path, source: StorageBackend) {
    copy_tree(from, to);
    for path in engine_paths(to, source) {
        if path.is_dir() {
            std::fs::remove_dir_all(&path).unwrap();
        } else if path.is_file() {
            std::fs::remove_file(&path).unwrap();
        }
    }
}

type State = BTreeMap<(String, Vec<u8>), Vec<u8>>;

/// Every row a migration carries for the tenant, as one engine stores it, with the
/// rollback sentinel and the keystore file. Derived index rows are left out.
fn raw_state(base: &Path, backend: StorageBackend) -> State {
    let dir = user_dir(base);
    let mut state = State::new();
    match backend {
        StorageBackend::RocksDb => {
            let storage = dir.join("storage");
            let opts = rocksdb::Options::default();
            let families = rocksdb::DB::list_cf(&opts, &storage).unwrap();
            let db = rocksdb::DB::open_cf_for_read_only(&opts, &storage, &families, false).unwrap();
            for family in ["default", "kv"] {
                let cf = db.cf_handle(family).unwrap();
                for item in db.iterator_cf(cf, rocksdb::IteratorMode::Start) {
                    let (k, v) = item.unwrap();
                    state.insert((family.to_string(), k.to_vec()), v.to_vec());
                }
            }
            let index = db.cf_handle("memory_index").unwrap();
            let sentinel = db.get_cf(index, b"meta:keystore_generation").unwrap();
            state.insert(
                ("sentinel".to_string(), Vec::new()),
                sentinel.expect("the rollback sentinel"),
            );
        }
        StorageBackend::Redb => {
            use redb::ReadableTable;
            let db = redb::Database::open(dir.join("storage.redb")).unwrap();
            let read = db.begin_read().unwrap();
            for table in ["default", "kv", "memories", "vector_mappings"] {
                let definition = redb::TableDefinition::<&[u8], &[u8]>::new(table);
                let Ok(handle) = read.open_table(definition) else {
                    continue;
                };
                for row in handle.iter().unwrap() {
                    let (k, v) = row.unwrap();
                    state.insert((table.to_string(), k.value().to_vec()), v.value().to_vec());
                }
            }
        }
    }
    state.insert(
        ("keystore".to_string(), Vec::new()),
        std::fs::read(keystore_path(base, backend)).unwrap(),
    );
    state
}

fn users() -> Vec<String> {
    vec![USER.to_string()]
}

fn assert_both_engines_boot(base: &Path) {
    for backend in [StorageBackend::Redb, StorageBackend::RocksDb] {
        let census = refuse_stranded_data(base, backend, &users())
            .unwrap_or_else(|e| panic!("{backend} refused to boot: {e:#}"));
        assert_eq!((census.rocksdb, census.redb), (1, Some(1)));
    }
}

fn assert_boot_refused(base: &Path, selected: StorageBackend, holder: StorageBackend) {
    let message = format!(
        "{:#}",
        refuse_stranded_data(base, selected, &users())
            .expect_err("an empty engine must not boot over the other's data")
    );
    for expected in [
        "refusing to serve".to_string(),
        USER.to_string(),
        format!("VELD_STORAGE_BACKEND={holder}"),
        format!("veld migrate --from {holder} --to {selected}"),
    ] {
        assert!(
            message.contains(&expected),
            "{expected:?} not in: {message}"
        );
    }
}

fn assert_opens_every_record(base: &Path, backend: StorageBackend, memories: &[Memory]) {
    fn check(store: &impl PrimaryMemoryStore, memories: &[Memory]) {
        assert_eq!(store.get_all().unwrap().len(), memories.len());
        for m in memories {
            assert_eq!(
                store.get(&m.id).unwrap().experience.content,
                m.experience.content
            );
            assert!(store.get_vector_mapping(&m.id).unwrap().is_some());
        }
    }
    let dir = user_dir(base);
    match backend {
        StorageBackend::RocksDb => check(
            &RocksDbPrimaryMemoryStore::open(&dir, None).unwrap(),
            memories,
        ),
        StorageBackend::Redb => check(&RedbPrimaryMemoryStore::open(&dir).unwrap(), memories),
    }
}

fn run(
    base: &Path,
    from: StorageBackend,
    to: StorageBackend,
    dry_run: bool,
) -> anyhow::Result<MigrateReport> {
    migrate::migrate(
        base,
        from,
        to,
        &Scope::User(USER.to_string()),
        dry_run,
        None,
    )
}

fn migrate_and_check_hop(base: &Path, from: StorageBackend, to: StorageBackend) {
    let source_before = bytes_at(&engine_paths(base, from));
    let report = run(base, from, to, false).unwrap_or_else(|e| panic!("{from} -> {to}: {e:#}"));
    let imported = &report.imported[0];
    assert_eq!(
        (imported.verified_frames, &imported.verified_digest),
        (imported.frames, &imported.digest),
        "{from} -> {to} did not verify"
    );
    assert_eq!(imported.verified_sealed_records, SEALED_COUNT);
    assert_eq!(
        bytes_at(&engine_paths(base, from)),
        source_before,
        "{from} -> {to} modified its source"
    );
    assert_eq!(
        leaks(&engine_paths(base, to)),
        Vec::<PathBuf>::new(),
        "{from} -> {to} left a sealed record readable on disk"
    );
}

fn encrypted_round_trip(origin_backend: StorageBackend) {
    let _env = Env::lock();
    let other = opposite(origin_backend);
    let tmp = TempDir::new().unwrap();
    let first = tmp.path().join("first");
    let second = tmp.path().join("second");
    let memories = encrypted_origin(&first, origin_backend);

    assert_boot_refused(&first, other, origin_backend);
    refuse_stranded_data(&first, origin_backend, &users()).expect("the holder boots");
    let origin = raw_state(&first, origin_backend);
    let origin_keystore = origin[&("keystore".to_string(), Vec::new())].clone();

    migrate_and_check_hop(&first, origin_backend, other);
    assert_eq!(
        std::fs::read(keystore_path(&first, other)).unwrap(),
        origin_keystore,
        "{origin_backend} -> {other}: the keystore did not travel as written"
    );
    carry_over_target_only(&first, &second, origin_backend);
    assert_both_engines_boot(&first);
    assert_opens_every_record(&first, origin_backend, &memories);
    assert_opens_every_record(&first, other, &memories);

    assert_boot_refused(&second, origin_backend, other);
    refuse_stranded_data(&second, other, &users()).expect("the migrated engine boots");

    migrate_and_check_hop(&second, other, origin_backend);
    assert_eq!(
        raw_state(&second, origin_backend),
        origin,
        "{origin_backend} -> {other} -> {origin_backend} did not return the origin's rows"
    );
    assert_both_engines_boot(&second);
    assert_opens_every_record(&second, origin_backend, &memories);
    assert_opens_every_record(&second, other, &memories);
}

#[test]
fn an_encrypted_rocksdb_tenant_returns_sealed_and_keyed_through_redb_and_back() {
    encrypted_round_trip(StorageBackend::RocksDb);
}

#[test]
fn an_encrypted_redb_tenant_returns_sealed_and_keyed_through_rocksdb_and_back() {
    encrypted_round_trip(StorageBackend::Redb);
}

#[test]
fn a_dry_run_of_an_encrypted_tenant_changes_no_byte_of_the_data_directory() {
    let _env = Env::lock();
    for from in [StorageBackend::RocksDb, StorageBackend::Redb] {
        let to = opposite(from);
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        encrypted_origin(base, from);
        let before = bytes_at(&[base.to_path_buf()]);

        let plan = run(base, from, to, true).expect("dry run");
        assert!(
            !plan.blocked(),
            "{from} -> {to}: {:?}",
            plan.units[0].blocked
        );
        assert_eq!(
            plan.units[0].source.as_ref().unwrap().sealed_records,
            SEALED_COUNT
        );
        assert!(plan.imported.is_empty());

        assert_eq!(
            bytes_at(&[base.to_path_buf()]),
            before,
            "{from} -> {to}: a dry run changed the data directory"
        );
    }
}

#[test]
fn a_target_holding_another_keystore_is_refused_by_name_in_both_directions() {
    let _env = Env::lock();
    for (from, to) in [
        (StorageBackend::RocksDb, StorageBackend::Redb),
        (StorageBackend::Redb, StorageBackend::RocksDb),
    ] {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        encrypted_origin(base, from);
        let planted_at = keystore_path(base, to);
        std::fs::create_dir_all(planted_at.parent().unwrap()).unwrap();
        Keystore::create(PASSPHRASE, KdfParams::production())
            .unwrap()
            .save_to_path(&planted_at)
            .unwrap();
        let before = bytes_at(&[base.to_path_buf()]);

        let plan = run(base, from, to, true).expect("dry run");
        assert!(
            plan.units[0].blocked.join("; ").contains("differs"),
            "{from} -> {to}: {:?}",
            plan.units[0].blocked
        );
        let message = format!("{:#}", run(base, from, to, false).expect_err("refused"));
        for expected in [
            "migration refused before writing anything",
            "already has a keystore",
            "differs from the source's",
            planted_at.to_str().unwrap(),
        ] {
            assert!(
                message.contains(expected),
                "{from} -> {to}: {expected:?} not in: {message}"
            );
        }
        assert_eq!(
            bytes_at(&[base.to_path_buf()]),
            before,
            "{from} -> {to}: a refused migration changed the data directory"
        );
    }
}

fn manager(dir: &Path, backend: StorageBackend) -> anyhow::Result<MultiUserMemoryManager> {
    let cfg = ServerConfig {
        storage_path: dir.to_path_buf(),
        backup_enabled: false,
        effective_storage_backend: backend,
        ..ServerConfig::default()
    };
    MultiUserMemoryManager::new(dir.to_path_buf(), cfg)
}

#[test]
fn the_real_manager_refuses_the_empty_rocksdb_engine_until_the_migration_completes() {
    let _env = Env::lock();
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();
    let memories = batch(25, PLAIN, "plain");
    write_memories(StorageBackend::Redb, &user_dir(base), &memories);

    let refusal = match manager(base, StorageBackend::RocksDb) {
        Ok(_) => panic!("a rocksdb server over redb-only data must not start"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        refusal.contains(USER) && refusal.contains("VELD_STORAGE_BACKEND=redb"),
        "{refusal}"
    );

    let source_before = bytes_at(&engine_paths(base, StorageBackend::Redb));
    run(base, StorageBackend::Redb, StorageBackend::RocksDb, false).expect("migrate");
    assert_eq!(
        bytes_at(&engine_paths(base, StorageBackend::Redb)),
        source_before
    );

    let booted = manager(base, StorageBackend::RocksDb)
        .expect("the migrated engine must boot once writes are stopped");
    assert_eq!(
        (booted.backend_census.rocksdb, booted.backend_census.redb),
        (1, Some(1))
    );
}
