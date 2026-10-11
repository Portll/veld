//! VELD_ENCRYPTION_KEY with no keystore refuses to open. It used to encrypt content;
//! now it would only decrypt, and every write after it would be plaintext.
#![cfg(feature = "rocksdb")]

use tempfile::TempDir;
use veld::memory::storage::MemoryStorage;

#[test]
fn a_legacy_key_without_a_keystore_refuses_to_open() {
    std::env::remove_var("VELD_MASTER_PASSPHRASE");
    std::env::remove_var("VELD_KMS_WRAP_KEY");
    std::env::set_var("VELD_ENCRYPTION_KEY", hex::encode([0x3c_u8; 32]));
    let temp = TempDir::new().unwrap();
    let err = MemoryStorage::new(temp.path(), None)
        .err()
        .expect("a legacy key alone must refuse");
    assert!(
        format!("{err:#}").contains("no keystore is active"),
        "{err:#}"
    );
    std::env::remove_var("VELD_ENCRYPTION_KEY");
}
