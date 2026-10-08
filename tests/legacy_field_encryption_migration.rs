//! Content veld v0.7.8–v0.7.18 field-encrypted under VELD_ENCRYPTION_KEY is opened and
//! re-sealed under the keystore by the read that finds it; without the key it is an
//! error, never ciphertext returned as the memory's text — including inside a record
//! that was sealed without its content being opened.

use aes_gcm::aead::{Aead, OsRng};
use aes_gcm::{AeadCore, Aes256Gcm, Key, KeyInit};
use base64::Engine;
use std::env;
use tempfile::TempDir;
use uuid::Uuid;
use veld::memory::storage::MemoryStorage;
use veld::memory::{Experience, ExperienceType, Memory, MemoryId};

const PASSPHRASE: &str = "legacy-migration-correct-horse-battery-staple-V7";
const LEGACY_KEY: [u8; 32] = [0x3c; 32];
const SECRET: &str = "veld-legacy-field-encrypted-distinctive-plaintext-do-not-leak-V7";

fn legacy_sealed(plain: &str) -> String {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&LEGACY_KEY));
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let mut out = b"ENC\x00".to_vec();
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&cipher.encrypt(&nonce, plain.as_bytes()).unwrap());
    base64::engine::general_purpose::STANDARD.encode(out)
}

fn memory_with(content: String) -> (MemoryId, Memory) {
    let id = MemoryId(Uuid::new_v4());
    let experience = Experience {
        experience_type: ExperienceType::Observation,
        content,
        ..Default::default()
    };
    (
        id.clone(),
        Memory::new(id, experience, 0.5, None, None, None, None),
    )
}

fn raw_record(temp: &TempDir, id: &MemoryId) -> Vec<u8> {
    let db_path = temp.path().join("storage");
    let opts = rocksdb::Options::default();
    let cfs = rocksdb::DB::list_cf(&opts, &db_path).unwrap();
    let db = rocksdb::DB::open_cf_for_read_only(&opts, &db_path, &cfs, false).unwrap();
    db.get(id.0.as_bytes()).unwrap().expect("record on disk")
}

#[test]
fn a_legacy_record_is_resealed_on_read_and_refused_without_its_key() {
    env::remove_var("VELD_MASTER_PASSPHRASE");
    env::remove_var("VELD_ENCRYPTION_KEY");
    let temp = TempDir::new().unwrap();
    let (legacy_id, legacy) = memory_with(legacy_sealed(SECRET));
    {
        // What a v0.7.x store holds: a plaintext record whose content is the field
        // ciphertext.
        let storage = MemoryStorage::new(temp.path(), None).expect("open plaintext store");
        storage.store(&legacy).expect("store");
    }

    env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);
    env::set_var("VELD_ENCRYPTION_KEY", hex::encode(LEGACY_KEY));
    let (inner_id, inner) = memory_with(legacy_sealed(SECRET));
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open with a keystore");
        assert!(storage.encrypts_records());
        assert_eq!(
            storage.get(&legacy_id).expect("opens").experience.content,
            SECRET
        );

        // Sealed WITHOUT its content opened, as the earlier record-level migration did.
        storage.store(&inner).expect("store sealed");
        env::remove_var("VELD_ENCRYPTION_KEY");
        let err = storage
            .get(&inner_id)
            .expect_err("legacy content inside an envelope, no key: an error");
        assert!(
            format!("{err:#}").contains("VELD_ENCRYPTION_KEY is unset"),
            "{err:#}"
        );
        assert_eq!(
            storage
                .get(&legacy_id)
                .expect("re-sealed")
                .experience
                .content,
            SECRET,
            "the migrated record no longer needs the legacy key"
        );
    }

    let raw = raw_record(&temp, &legacy_id);
    assert!(raw.starts_with(b"ENC\0"), "the read re-sealed the record");
    assert!(!raw.windows(SECRET.len()).any(|w| w == SECRET.as_bytes()));
    env::remove_var("VELD_MASTER_PASSPHRASE");
}
