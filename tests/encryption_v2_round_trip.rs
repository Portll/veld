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
