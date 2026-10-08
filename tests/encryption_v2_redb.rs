#![cfg(feature = "storage-redb")]
//! Record-level encryption (encryption-v2) on the redb engine: the envelope, keystore
//! rules and rollback guard RocksDB applies, applied to `storage.redb`.
//!
//! Own test binary, because setting `VELD_MASTER_PASSPHRASE` is process-global.

use std::env;
use std::path::Path;
use std::sync::Mutex;

use tempfile::TempDir;
use uuid::Uuid;

use veld::config::StorageBackend;
use veld::keystore::Keystore;
use veld::memory::{Experience, ExperienceType, Memory, MemoryId};
use veld::storage::migrate::{self, Scope};
use veld::storage::redb::{RedbPrimaryMemoryStore, KEYSTORE_FILE};
use veld::storage::PrimaryMemoryStore;

static ENV_LOCK: Mutex<()> = Mutex::new(());

const PASSPHRASE: &str = "redb-v2-correct-horse-battery-staple-Q7";
const PLAINTEXT: &str = "redb-v2-record-encryption-distinctive-plaintext-do-not-leak-Q7X";
const USER: &str = "alice";

fn sample(content: &str) -> (MemoryId, Memory) {
    let id = MemoryId(Uuid::new_v4());
    let experience = Experience {
        experience_type: ExperienceType::Observation,
        content: content.to_string(),
        ..Default::default()
    };
    (
        id.clone(),
        Memory::new(id, experience, 0.5, None, None, None, None),
    )
}

fn contains(haystack: &[u8], needle: &str) -> bool {
    haystack
        .windows(needle.len())
        .any(|w| w == needle.as_bytes())
}

fn user_dir(base: &Path) -> std::path::PathBuf {
    let dir = base.join(USER);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn content(store: &RedbPrimaryMemoryStore, id: &MemoryId) -> String {
    store.get(id).expect("get").experience.content
}

#[test]
fn a_record_round_trips_and_only_ciphertext_reaches_the_file() {
    let _guard = ENV_LOCK.lock().unwrap();
    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);
    let temp = TempDir::new().unwrap();
    let dir = user_dir(temp.path());
    let (id, memory) = sample(PLAINTEXT);

    {
        let store = RedbPrimaryMemoryStore::open(&dir).expect("open");
        store.store(&memory).expect("store");
        assert_eq!(content(&store, &id), PLAINTEXT);
        assert_eq!(store.get_all().expect("get_all").len(), 1);
    }

    assert!(
        dir.join(KEYSTORE_FILE).exists(),
        "the first open with a passphrase creates the keystore"
    );
    let file = std::fs::read(dir.join("storage.redb")).unwrap();
    assert!(
        !contains(&file, PLAINTEXT),
        "plaintext reached storage.redb"
    );
    env::remove_var("VELD_MASTER_PASSPHRASE");
}

#[test]
fn the_wrong_passphrase_and_a_missing_one_both_refuse_to_open() {
    let _guard = ENV_LOCK.lock().unwrap();
    let temp = TempDir::new().unwrap();
    let dir = user_dir(temp.path());

    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);
    {
        let store = RedbPrimaryMemoryStore::open(&dir).expect("create keystore");
        store.store(&sample("x").1).expect("store");
    }

    env::set_var("VELD_MASTER_PASSPHRASE", "the-wrong-passphrase");
    assert!(
        RedbPrimaryMemoryStore::open(&dir).is_err(),
        "wrong passphrase opened the store"
    );

    env::remove_var("VELD_MASTER_PASSPHRASE");
    env::remove_var("VELD_KMS_WRAP_KEY");
    assert!(
        RedbPrimaryMemoryStore::open(&dir).is_err(),
        "a keystore with no passphrase would serve ciphertext as plaintext"
    );
}

#[test]
fn records_written_before_the_keystore_stay_readable_beside_sealed_ones() {
    let _guard = ENV_LOCK.lock().unwrap();
    let temp = TempDir::new().unwrap();
    let dir = user_dir(temp.path());
    let (plain_id, plain) = sample("written-in-plaintext-before-encryption");
    let (sealed_id, sealed) = sample(PLAINTEXT);

    env::remove_var("VELD_MASTER_PASSPHRASE");
    {
        let store = RedbPrimaryMemoryStore::open(&dir).expect("plaintext open");
        store.store(&plain).expect("store plaintext");
    }
    assert!(!dir.join(KEYSTORE_FILE).exists());

    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);
    {
        let store = RedbPrimaryMemoryStore::open(&dir).expect("encrypted open");
        store.store(&sealed).expect("store sealed");
        assert_eq!(
            content(&store, &plain_id),
            "written-in-plaintext-before-encryption"
        );
        assert_eq!(content(&store, &sealed_id), PLAINTEXT);
        assert_eq!(
            store.cleanup_corrupted().expect("cleanup"),
            0,
            "nothing is corrupt"
        );
    }
    let file = std::fs::read(dir.join("storage.redb")).unwrap();
    assert!(!contains(&file, PLAINTEXT));
    env::remove_var("VELD_MASTER_PASSPHRASE");
}

#[test]
fn a_rotated_key_keeps_old_records_readable_and_an_older_keystore_is_refused() {
    let _guard = ENV_LOCK.lock().unwrap();
    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);
    env::remove_var("VELD_ALLOW_KEYSTORE_ROLLBACK");
    let temp = TempDir::new().unwrap();
    let dir = user_dir(temp.path());
    let keystore_path = dir.join(KEYSTORE_FILE);
    let (old_id, old) = sample("before-rotation");

    {
        let store = RedbPrimaryMemoryStore::open(&dir).expect("open");
        store.store(&old).expect("store");
    }
    let before_rotation = std::fs::read(&keystore_path).unwrap();

    {
        let mut ks =
            Keystore::from_json(&String::from_utf8(before_rotation.clone()).unwrap()).unwrap();
        let kek = ks.unseal_with_passphrase(PASSPHRASE).unwrap();
        assert_eq!(ks.rotate_dek(&kek).unwrap(), 1);
        ks.save_to_path(&keystore_path).unwrap();
    }

    let (new_id, new) = sample("after-rotation");
    {
        let store = RedbPrimaryMemoryStore::open(&dir).expect("reopen after rotation");
        assert_eq!(
            content(&store, &old_id),
            "before-rotation",
            "epoch-0 record unreadable after rotation"
        );
        store.store(&new).expect("store under epoch 1");
        assert_eq!(content(&store, &new_id), "after-rotation");
    }

    std::fs::write(&keystore_path, &before_rotation).unwrap();
    let err = RedbPrimaryMemoryStore::open(&dir)
        .err()
        .expect("an older keystore generation must be refused");
    assert!(format!("{err:#}").contains("rollback"), "{err:#}");
    env::remove_var("VELD_MASTER_PASSPHRASE");
}

#[test]
fn an_export_carries_sealed_records_sealed() {
    let _guard = ENV_LOCK.lock().unwrap();
    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);
    let temp = TempDir::new().unwrap();
    let dir = user_dir(temp.path());
    let (_id, memory) = sample(PLAINTEXT);
    {
        let store = RedbPrimaryMemoryStore::open(&dir).expect("open");
        store.store(&memory).expect("store");
    }

    let out = temp.path().join("export");
    let reports = migrate::export(
        temp.path(),
        StorageBackend::Redb,
        &Scope::User(USER.to_string()),
        &out,
        None,
    )
    .expect("export of an encrypted redb store");
    assert_eq!(reports.len(), 1);
    assert!(reports[0].records >= 1);
    let dump = std::fs::read(&reports[0].path).unwrap();
    assert!(!contains(&dump, PLAINTEXT), "the export decrypted a record");

    env::remove_var("VELD_MASTER_PASSPHRASE");
}
