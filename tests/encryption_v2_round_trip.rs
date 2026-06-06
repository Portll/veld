//! End-to-end record-level encryption (encryption-v2, P2) against a real RocksDB.
//!
//! Runs in its own test binary so setting `VELD_MASTER_PASSPHRASE` cannot leak
//! into other storage tests. Verifies:
//!   1. with a passphrase, a stored memory round-trips through `store()`/`get()`;
//!   2. on disk the record is an opaque `ENC\0` envelope with the plaintext absent;
//!   3. reopening with the wrong passphrase is a hard error (never silent plaintext).

use std::env;
use std::sync::Mutex;

use tempfile::TempDir;
use uuid::Uuid;

use veld::memory::storage::MemoryStorage;
use veld::memory::{Experience, ExperienceType, Memory, MemoryId};

/// `env::set_var`/`remove_var` are process-global; serialize the env-touching tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

const PASSPHRASE: &str = "rT4-v2-e2e-correct-horse-battery-staple-Z9";
const PLAINTEXT: &str = "rT4-v2-record-encryption-distinctive-plaintext-do-not-leak-Z9X-2026";

fn sample(content: &str) -> (MemoryId, Memory) {
    let id = MemoryId(Uuid::new_v4());
    let experience = Experience {
        experience_type: ExperienceType::Observation,
        content: content.to_string(),
        ..Default::default()
    };
    let memory = Memory::new(id.clone(), experience, 0.5, None, None, None, None);
    (id, memory)
}

#[test]
fn record_encryption_round_trip_against_real_rocksdb() {
    let _guard = ENV_LOCK.lock().unwrap();
    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);

    let temp = TempDir::new().expect("temp dir");
    let (id, memory) = sample(PLAINTEXT);

    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open storage");
        storage.store(&memory).expect("store");

        // Round-trips through the decrypt-on-read path.
        let got = storage.get(&id).expect("get");
        assert_eq!(got.experience.content, PLAINTEXT);
    } // drop storage -> release the RocksDB lock for the raw read below

    // Raw on-disk check: stored record must be an opaque ENC\0 envelope, no plaintext.
    let db_path = temp.path().join("storage");
    let opts = rocksdb::Options::default();
    let cf_names = rocksdb::DB::list_cf(&opts, &db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, &db_path, &cf_names, false)
        .expect("reopen read-only");
    let raw = db
        .get(id.0.as_bytes())
        .expect("rocksdb get")
        .expect("record present on disk");

    assert!(
        raw.starts_with(b"ENC\0"),
        "stored record must be an opaque record-level ciphertext"
    );
    assert!(
        !raw.windows(PLAINTEXT.len())
            .any(|w| w == PLAINTEXT.as_bytes()),
        "plaintext must NOT appear in the stored bytes (raw {} bytes)",
        raw.len()
    );

    env::remove_var("VELD_MASTER_PASSPHRASE");
}

#[test]
fn index_terms_are_blinded_on_disk() {
    let _guard = ENV_LOCK.lock().unwrap();
    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);

    let temp = TempDir::new().expect("temp dir");
    const SECRET_TAG: &str = "secret-tag-do-not-leak-abc123";
    const SECRET_ENTITY: &str = "secret-entity-do-not-leak-def456";

    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open storage");
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: "indexed content".to_string(),
            tags: vec![SECRET_TAG.to_string()],
            entities: vec![SECRET_ENTITY.to_string()],
            ..Default::default()
        };
        let memory = Memory::new(MemoryId(Uuid::new_v4()), experience, 0.5, None, None, None, None);
        storage.store(&memory).expect("store");
    }

    // Scan the memory_index CF: no plaintext tag/entity term may appear in any key.
    let db_path = temp.path().join("storage");
    let opts = rocksdb::Options::default();
    let cf_names = rocksdb::DB::list_cf(&opts, &db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, &db_path, &cf_names, false)
        .expect("reopen read-only");
    let cf = db.cf_handle("memory_index").expect("memory_index cf");

    let mut saw_tag_prefix = false;
    for item in db.iterator_cf(&cf, rocksdb::IteratorMode::Start) {
        let (key, _) = item.expect("iter");
        let ks = String::from_utf8_lossy(&key);
        assert!(!ks.contains(SECRET_TAG), "plaintext tag leaked in index key: {ks}");
        assert!(
            !ks.contains(SECRET_ENTITY),
            "plaintext entity leaked in index key: {ks}"
        );
        if ks.starts_with("tag:") {
            saw_tag_prefix = true;
        }
    }
    assert!(saw_tag_prefix, "expected a blinded `tag:` index entry to exist");

    env::remove_var("VELD_MASTER_PASSPHRASE");
}

#[test]
fn rebuild_index_is_record_safe() {
    let _guard = ENV_LOCK.lock().unwrap();
    env::remove_var("VELD_MASTER_PASSPHRASE"); // plaintext path: fast, no Argon2

    let temp = TempDir::new().expect("temp dir");
    let storage = MemoryStorage::new(temp.path(), None).expect("open storage");
    let (id1, m1) = sample("alpha");
    let (_id2, m2) = sample("beta");
    storage.store(&m1).expect("store m1");
    storage.store(&m2).expect("store m2");

    let n = storage.rebuild_index().expect("rebuild");
    assert_eq!(n, 2, "rebuild should re-derive all records");
    // Records survive the rebuild unchanged.
    assert!(storage.get(&id1).is_ok(), "record must survive index rebuild");
}

#[test]
fn wrong_passphrase_after_keystore_created_is_hard_error() {
    let _guard = ENV_LOCK.lock().unwrap();
    let temp = TempDir::new().expect("temp dir");

    // First open creates the keystore under the correct passphrase and stores a record.
    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("create keystore");
        let (_id, m) = sample("x");
        storage.store(&m).expect("store");
    }

    // Reopen with the WRONG passphrase: unseal must fail rather than serve ciphertext.
    env::set_var("VELD_MASTER_PASSPHRASE", "the-wrong-passphrase");
    assert!(
        MemoryStorage::new(temp.path(), None).is_err(),
        "opening with the wrong passphrase must be a hard error"
    );

    env::remove_var("VELD_MASTER_PASSPHRASE");
}

#[test]
fn dek_rotation_keeps_old_records_readable() {
    let _guard = ENV_LOCK.lock().unwrap();
    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);

    let temp = TempDir::new().expect("temp dir");
    let (old_id, old_mem) = sample("record-written-before-rotation");

    // 1. Create the keystore (epoch 0) and store a record under it.
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open storage");
        storage.store(&old_mem).expect("store pre-rotation record");
        assert_eq!(
            storage.get(&old_id).expect("get pre-rotation").experience.content,
            "record-written-before-rotation"
        );
    }

    // 2. Rotate the data key out-of-process, exactly as `veld-keyctl rotate-dek`:
    //    load the keystore, unseal, rotate the DEK to a new epoch, persist.
    let keystore_path = temp.path().join("storage").join("keystore.json");
    {
        use veld::keystore::Keystore;
        let json = std::fs::read_to_string(&keystore_path).expect("read keystore");
        let mut ks = Keystore::from_json(&json).expect("parse keystore");
        let kek = ks.unseal_with_passphrase(PASSPHRASE).expect("unseal");
        let new_epoch = ks.rotate_dek(&kek).expect("rotate dek");
        assert_eq!(new_epoch, 1, "rotation should produce epoch 1");
        ks.save_to_path(&keystore_path).expect("persist rotated keystore");
    }

    // 3. Reopen: the pre-rotation record (epoch 0) must still decrypt via its
    //    retained DEK, and a new record is written under the active epoch 1.
    let (new_id, new_mem) = sample("record-written-after-rotation");
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("reopen after rotation");
        assert_eq!(
            storage.get(&old_id).expect("read old record after rotation").experience.content,
            "record-written-before-rotation",
            "pre-rotation record must remain readable under its retired epoch key"
        );
        storage.store(&new_mem).expect("store post-rotation record");
        assert_eq!(
            storage.get(&new_id).expect("read new record").experience.content,
            "record-written-after-rotation"
        );
    }

    // 4. On disk the two records carry different epochs in the envelope header
    //    (ENC\0 marker[4] | version[1] | epoch[4 LE] | nonce | ct): old=0, new=1.
    let db_path = temp.path().join("storage");
    let opts = rocksdb::Options::default();
    let cf_names = rocksdb::DB::list_cf(&opts, &db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, &db_path, &cf_names, false)
        .expect("reopen read-only");
    let epoch_of = |id: &MemoryId| -> u32 {
        let raw = db.get(id.0.as_bytes()).expect("get").expect("present");
        assert!(raw.starts_with(b"ENC\0"), "record must be an ENC\\0 envelope");
        u32::from_le_bytes([raw[5], raw[6], raw[7], raw[8]])
    };
    assert_eq!(epoch_of(&old_id), 0, "pre-rotation record should be epoch 0");
    assert_eq!(epoch_of(&new_id), 1, "post-rotation record should be epoch 1");

    env::remove_var("VELD_MASTER_PASSPHRASE");
}
