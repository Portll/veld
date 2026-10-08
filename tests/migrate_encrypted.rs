#![cfg(all(feature = "rocksdb", feature = "storage-redb"))]
//! `veld migrate` between the engines for a tenant with record-level encryption: sealed
//! records arrive as stored, the keystore and its rollback sentinel arrive with them, and
//! a store whose keys are not at hand is refused before anything is written.
//!
//! Own test binary, because `VELD_MASTER_PASSPHRASE` is process-global.

use std::env;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tempfile::TempDir;
use uuid::Uuid;

use veld::config::StorageBackend;
use veld::keystore::{Keystore, KmsClient, LocalAeadKms};
use veld::memory::storage::{MemoryStorage, SearchCriteria};
use veld::memory::{Experience, ExperienceType, Memory, MemoryId};
use veld::storage::migrate::{self, MigrateReport, Scope};
use veld::storage::redb::RedbPrimaryMemoryStore;
use veld::storage::PrimaryMemoryStore;

static ENV_LOCK: Mutex<()> = Mutex::new(());

const PASSPHRASE: &str = "migrate-sealed-correct-horse-battery-staple-M3";
const USER: &str = "nadia";
const SECRET_BEFORE: &str = "sealed-before-rotation-do-not-leak-M3Q";
const SECRET_AFTER: &str = "sealed-after-rotation-do-not-leak-M3W";
const SECRET_TAG: &str = "secret-tag-do-not-leak-M3T";
const PLAIN: &str = "written-in-plaintext-before-the-keystore";
const KMS_KEY: &str = "0f1e2d3c4b5a69788796a5b4c3d2e1f00112233445566778899aabbccddeeff0";

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

/// The memories a source holds: one plaintext, one sealed at epoch 0, one at epoch 1.
struct Source {
    plain: Memory,
    before: Memory,
    after: Memory,
    /// The keystore as it was before the rotation (generation 0).
    old_keystore: Vec<u8>,
}

impl Source {
    fn all(&self) -> [&Memory; 3] {
        [&self.plain, &self.before, &self.after]
    }
}

fn rotate(keystore: &Path) {
    let mut ks = Keystore::from_json(&std::fs::read_to_string(keystore).unwrap()).unwrap();
    let kek = ks.unseal_with_passphrase(PASSPHRASE).unwrap();
    assert_eq!(ks.rotate_dek(&kek).unwrap(), 1);
    ks.save_to_path(keystore).unwrap();
}

fn rocks_source(base: &Path) -> Source {
    let dir = base.join(USER);
    let keystore = dir.join("storage/keystore.json");
    let source = Source {
        plain: sample(PLAIN, &["plain"]),
        before: sample(SECRET_BEFORE, &[SECRET_TAG]),
        after: sample(SECRET_AFTER, &["after"]),
        old_keystore: Vec::new(),
    };
    MemoryStorage::new(&dir, None)
        .unwrap()
        .store(&source.plain)
        .unwrap();
    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);
    MemoryStorage::new(&dir, None)
        .unwrap()
        .store(&source.before)
        .unwrap();
    let old_keystore = std::fs::read(&keystore).unwrap();
    rotate(&keystore);
    MemoryStorage::new(&dir, None)
        .unwrap()
        .store(&source.after)
        .unwrap();
    Source {
        old_keystore,
        ..source
    }
}

fn redb_source(base: &Path) -> Source {
    let dir = base.join(USER);
    std::fs::create_dir_all(&dir).unwrap();
    let keystore = dir.join("keystore.json");
    let source = Source {
        plain: sample(PLAIN, &["plain"]),
        before: sample(SECRET_BEFORE, &[SECRET_TAG]),
        after: sample(SECRET_AFTER, &["after"]),
        old_keystore: Vec::new(),
    };
    RedbPrimaryMemoryStore::open(&dir)
        .unwrap()
        .store(&source.plain)
        .unwrap();
    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);
    RedbPrimaryMemoryStore::open(&dir)
        .unwrap()
        .store(&source.before)
        .unwrap();
    let old_keystore = std::fs::read(&keystore).unwrap();
    rotate(&keystore);
    RedbPrimaryMemoryStore::open(&dir)
        .unwrap()
        .store(&source.after)
        .unwrap();
    Source {
        old_keystore,
        ..source
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

/// Memory rows by key.
type Memories = std::collections::BTreeMap<Vec<u8>, Vec<u8>>;

/// A store's memories and its rollback sentinel.
fn redb_raw(dir: &Path) -> (Memories, Option<Vec<u8>>) {
    use redb::ReadableTable;
    let db = redb::Database::open(dir.join("storage.redb")).unwrap();
    let read = db.begin_read().unwrap();
    let raw = redb::TableDefinition::<&[u8], &[u8]>::new;
    let memories = read
        .open_table(raw("memories"))
        .unwrap()
        .iter()
        .unwrap()
        .map(|row| {
            let (k, v) = row.unwrap();
            (k.value().to_vec(), v.value().to_vec())
        })
        .collect();
    let sentinel = read
        .open_table(raw("default"))
        .unwrap()
        .get(b"meta:keystore_generation".as_slice())
        .unwrap()
        .map(|v| v.value().to_vec());
    (memories, sentinel)
}

type Rows = Vec<(Vec<u8>, Vec<u8>)>;

/// Every row of every family of a rocksdb store, its memories, and the rollback sentinel.
fn rocks_raw(storage: &Path) -> (Rows, Memories, Option<Vec<u8>>) {
    let opts = rocksdb::Options::default();
    let families = rocksdb::DB::list_cf(&opts, storage).unwrap();
    let db = rocksdb::DB::open_cf_for_read_only(&opts, storage, &families, false).unwrap();
    let mut rows = Vec::new();
    for family in &families {
        let cf = db.cf_handle(family).unwrap();
        for item in db.iterator_cf(cf, rocksdb::IteratorMode::Start) {
            let (k, v) = item.unwrap();
            rows.push((k.to_vec(), v.to_vec()));
        }
    }
    let memories = rows
        .iter()
        .filter(|(k, _)| k.len() == 16)
        .cloned()
        .collect();
    let index = db.cf_handle("memory_index").unwrap();
    let sentinel = db.get_cf(index, b"meta:keystore_generation").unwrap();
    (rows, memories, sentinel)
}

fn assert_sealed_as_stored(source: &Source, from: &Memories, to: &Memories) {
    for m in source.all() {
        let key = m.id.0.as_bytes().to_vec();
        assert_eq!(
            from.get(&key),
            to.get(&key),
            "{} changed in transit",
            m.id.0
        );
    }
    let epoch = |m: &Memory| {
        let raw = &to[&m.id.0.as_bytes().to_vec()];
        assert!(raw.starts_with(b"ENC\0"), "{} arrived unsealed", m.id.0);
        u32::from_le_bytes([raw[5], raw[6], raw[7], raw[8]])
    };
    assert_eq!(epoch(&source.before), 0);
    assert_eq!(epoch(&source.after), 1);
    assert!(!to[&source.plain.id.0.as_bytes().to_vec()].starts_with(b"ENC\0"));
}

#[test]
fn rocksdb_to_redb_carries_sealed_records_the_keystore_and_its_sentinel() {
    let _env = Env::lock();
    let temp = TempDir::new().unwrap();
    let base = temp.path();
    let source = rocks_source(base);
    let dir = base.join(USER);
    let (_, rocks_memories, _) = rocks_raw(&dir.join("storage"));

    let plan = run(base, StorageBackend::RocksDb, StorageBackend::Redb, true).expect("dry run");
    assert!(!plan.blocked(), "{:?}", plan.units[0].blocked);
    assert_eq!(plan.units[0].source.as_ref().unwrap().sealed_records, 2);
    assert!(
        !dir.join("keystore.json").exists(),
        "the dry run copied the keystore"
    );
    assert!(
        !dir.join("storage.redb").exists(),
        "the dry run wrote a target"
    );

    let report = run(base, StorageBackend::RocksDb, StorageBackend::Redb, false).expect("migrate");
    let imported = &report.imported[0];
    assert_eq!(imported.verified_sealed_records, 2);
    assert_eq!(
        (imported.verified_frames, &imported.verified_digest),
        (imported.frames, &imported.digest)
    );

    let copied = std::fs::read(dir.join("keystore.json")).unwrap();
    assert_eq!(
        copied,
        std::fs::read(dir.join("storage/keystore.json")).unwrap()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(dir.join("keystore.json"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o077, 0, "the copied keystore is readable by others");
    }
    let (redb_memories, sentinel) = redb_raw(&dir);
    assert_sealed_as_stored(&source, &rocks_memories, &redb_memories);
    assert_eq!(sentinel, Some(1u64.to_le_bytes().to_vec()));
    let targets = [
        dir.join("storage.redb"),
        dir.join("graph/graph.redb"),
        dir.join("keystore.json"),
    ];
    assert_eq!(leaks(&targets), Vec::<PathBuf>::new());

    {
        let redb = RedbPrimaryMemoryStore::open(&dir).expect("open the migrated store");
        for m in source.all() {
            assert_eq!(
                redb.get(&m.id).unwrap().experience.content,
                m.experience.content
            );
        }
    }

    std::fs::write(dir.join("keystore.json"), &source.old_keystore).unwrap();
    let err = RedbPrimaryMemoryStore::open(&dir)
        .err()
        .expect("the migrated sentinel refuses an older keystore");
    assert!(format!("{err:#}").contains("rollback"), "{err:#}");
}

#[test]
fn redb_to_rocksdb_carries_sealed_records_and_rebuilds_the_blinded_index() {
    let _env = Env::lock();
    let temp = TempDir::new().unwrap();
    let base = temp.path();
    let source = redb_source(base);
    let dir = base.join(USER);
    let (redb_memories, _) = redb_raw(&dir);

    let plan = run(base, StorageBackend::Redb, StorageBackend::RocksDb, true).expect("dry run");
    assert!(!plan.blocked(), "{:?}", plan.units[0].blocked);
    assert!(!dir.join("storage").exists(), "the dry run wrote a target");

    let report = run(base, StorageBackend::Redb, StorageBackend::RocksDb, false).expect("migrate");
    assert_eq!(report.imported[0].verified_sealed_records, 2);

    let storage = dir.join("storage");
    assert_eq!(
        std::fs::read(storage.join("keystore.json")).unwrap(),
        std::fs::read(dir.join("keystore.json")).unwrap()
    );
    let (rows, rocks_memories, sentinel) = rocks_raw(&storage);
    assert_sealed_as_stored(&source, &redb_memories, &rocks_memories);
    assert_eq!(sentinel, Some(1u64.to_le_bytes().to_vec()));
    for (k, v) in &rows {
        for secret in [SECRET_BEFORE, SECRET_AFTER, SECRET_TAG] {
            assert!(
                !contains(k, secret) && !contains(v, secret),
                "a row holds {secret}"
            );
        }
    }
    assert_eq!(
        leaks(&[storage.clone(), dir.join("graph/graph")]),
        Vec::<PathBuf>::new()
    );

    {
        let rocks = MemoryStorage::new(&dir, None).expect("open the migrated store");
        for m in source.all() {
            assert_eq!(
                rocks.get(&m.id).unwrap().experience.content,
                m.experience.content
            );
        }
        let tagged = rocks
            .search(SearchCriteria::ByTags(vec![SECRET_TAG.to_string()]))
            .unwrap();
        assert_eq!(
            tagged.iter().map(|m| m.id.0).collect::<Vec<_>>(),
            vec![source.before.id.0],
            "the index was not rebuilt with the copied keystore's blinder"
        );
    }

    std::fs::write(storage.join("keystore.json"), &source.old_keystore).unwrap();
    assert!(
        MemoryStorage::new(&dir, None).is_err(),
        "the migrated sentinel refuses an older keystore"
    );
}

#[test]
fn an_encrypted_store_without_its_keys_is_refused_and_nothing_is_written() {
    let _env = Env::lock();
    for (from, to) in [
        (StorageBackend::RocksDb, StorageBackend::Redb),
        (StorageBackend::Redb, StorageBackend::RocksDb),
    ] {
        clear_env();
        let temp = TempDir::new().unwrap();
        let base = temp.path();
        match from {
            StorageBackend::RocksDb => rocks_source(base),
            StorageBackend::Redb => redb_source(base),
        };
        let dir = base.join(USER);
        let target_paths = match to {
            StorageBackend::Redb => vec![dir.join("storage.redb"), dir.join("keystore.json")],
            StorageBackend::RocksDb => vec![dir.join("storage")],
        };

        env::remove_var("VELD_MASTER_PASSPHRASE");
        let plan = run(base, from, to, true).expect("dry run");
        let blocked = plan.units[0].blocked.join("; ");
        assert!(
            blocked.contains("VELD_MASTER_PASSPHRASE") && blocked.contains("VELD_KMS_WRAP_KEY"),
            "{from} -> {to}: {blocked}"
        );
        let err = run(base, from, to, false).expect_err("no keys");
        assert!(
            format!("{err:#}").contains("VELD_MASTER_PASSPHRASE"),
            "{err:#}"
        );

        env::set_var("VELD_MASTER_PASSPHRASE", "not-the-passphrase");
        let plan = run(base, from, to, true).expect("dry run");
        assert!(
            plan.units[0].blocked.join("; ").contains("does not unseal"),
            "{from} -> {to}: {:?}",
            plan.units[0].blocked
        );
        assert!(
            run(base, from, to, false).is_err(),
            "{from} -> {to}: wrong passphrase"
        );

        for path in &target_paths {
            assert!(
                !path.exists(),
                "{from} -> {to}: {} was written",
                path.display()
            );
        }
    }
}

#[test]
fn a_target_holding_another_keystore_is_refused_and_an_identical_one_is_kept() {
    let _env = Env::lock();
    let temp = TempDir::new().unwrap();
    let base = temp.path();
    let source = rocks_source(base);
    let dir = base.join(USER);

    let other = Keystore::create(PASSPHRASE, veld::keystore::KdfParams::production()).unwrap();
    other.save_to_path(&dir.join("keystore.json")).unwrap();
    let planted = std::fs::read(dir.join("keystore.json")).unwrap();
    let plan = run(base, StorageBackend::RocksDb, StorageBackend::Redb, true).expect("dry run");
    assert!(
        plan.units[0].blocked.join("; ").contains("differs"),
        "{:?}",
        plan.units[0].blocked
    );
    assert!(run(base, StorageBackend::RocksDb, StorageBackend::Redb, false).is_err());
    assert_eq!(std::fs::read(dir.join("keystore.json")).unwrap(), planted);
    assert!(!dir.join("storage.redb").exists());

    std::fs::copy(dir.join("storage/keystore.json"), dir.join("keystore.json")).unwrap();
    run(base, StorageBackend::RocksDb, StorageBackend::Redb, false).expect("an identical keystore");
    let redb = RedbPrimaryMemoryStore::open(&dir).unwrap();
    assert_eq!(
        redb.get(&source.after.id).unwrap().experience.content,
        SECRET_AFTER
    );
}

#[test]
fn a_source_whose_keystore_is_older_than_its_sentinel_is_refused() {
    let _env = Env::lock();
    let temp = TempDir::new().unwrap();
    let base = temp.path();
    let source = rocks_source(base);
    let dir = base.join(USER);
    std::fs::write(dir.join("storage/keystore.json"), &source.old_keystore).unwrap();

    let plan = run(base, StorageBackend::RocksDb, StorageBackend::Redb, true).expect("dry run");
    assert!(
        plan.units[0].blocked.join("; ").contains("rollback"),
        "{:?}",
        plan.units[0].blocked
    );
    assert!(run(base, StorageBackend::RocksDb, StorageBackend::Redb, false).is_err());
    assert!(!dir.join("storage.redb").exists());
}

#[test]
fn a_kms_wrap_key_unseals_the_keystore_when_no_passphrase_is_set() {
    let _env = Env::lock();
    let temp = TempDir::new().unwrap();
    let base = temp.path();
    let source = redb_source(base);
    let dir = base.join(USER);
    let keystore = dir.join("keystore.json");
    {
        let mut ks = Keystore::from_json(&std::fs::read_to_string(&keystore).unwrap()).unwrap();
        let kek = ks.unseal_with_passphrase(PASSPHRASE).unwrap();
        let kms = LocalAeadKms::new(hex_key(KMS_KEY));
        ks.add_kms_wrap(&kek, &kms as &dyn KmsClient).unwrap();
        ks.save_to_path(&keystore).unwrap();
    }
    env::remove_var("VELD_MASTER_PASSPHRASE");
    env::set_var("VELD_KMS_WRAP_KEY", KMS_KEY);

    let report = run(base, StorageBackend::Redb, StorageBackend::RocksDb, false).expect("migrate");
    assert_eq!(report.imported[0].verified_sealed_records, 2);
    let rocks = MemoryStorage::new(&dir, None).expect("open under the KMS key");
    for m in source.all() {
        assert_eq!(
            rocks.get(&m.id).unwrap().experience.content,
            m.experience.content
        );
    }
}

fn hex_key(text: &str) -> [u8; 32] {
    let mut key = [0u8; 32];
    for (i, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).unwrap();
    }
    key
}

#[test]
fn a_plaintext_store_migrates_while_a_passphrase_is_set_and_stays_plaintext() {
    let _env = Env::lock();
    for (from, to) in [
        (StorageBackend::Redb, StorageBackend::RocksDb),
        (StorageBackend::RocksDb, StorageBackend::Redb),
    ] {
        let temp = TempDir::new().unwrap();
        let base = temp.path();
        let dir = base.join(USER);
        std::fs::create_dir_all(&dir).unwrap();
        let memory = sample(PLAIN, &["plain"]);
        match from {
            StorageBackend::Redb => RedbPrimaryMemoryStore::open(&dir)
                .unwrap()
                .store(&memory)
                .unwrap(),
            StorageBackend::RocksDb => MemoryStorage::new(&dir, None)
                .unwrap()
                .store(&memory)
                .unwrap(),
        }
        env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);

        let report = run(base, from, to, false)
            .unwrap_or_else(|e| panic!("{from} -> {to} with a passphrase set: {e:#}"));
        assert_eq!(
            report.imported[0].verified_sealed_records, 0,
            "{from} -> {to}"
        );
        let target_keystore = match to {
            StorageBackend::RocksDb => dir.join("storage/keystore.json"),
            StorageBackend::Redb => dir.join("keystore.json"),
        };
        assert!(
            !target_keystore.exists(),
            "{from} -> {to}: migrate created a keystore on the plaintext target"
        );
        env::remove_var("VELD_MASTER_PASSPHRASE");
    }
}
