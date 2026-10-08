//! With only VELD_MASTER_PASSPHRASE set, the records are sealed under the store's
//! keystore; the journal beside them carries the same memories, so its key is derived
//! from that keystore too — one keystore, created once, whichever opens it first.

use tempfile::TempDir;
use veld::keystore::{open_record_keys, SecretCryptor};

#[test]
fn the_journal_and_the_records_share_one_store_keystore() {
    std::env::remove_var("VELD_KEYSTORE_PATH");
    std::env::set_var(
        "VELD_MASTER_PASSPHRASE",
        "journal-shares-keystore-correct-horse-J4",
    );
    let dir = TempDir::new().unwrap();
    // A fresh tenant: the journal opens first, before the store made its directory.
    let path = dir.path().join("storage").join("keystore.json");

    let journal = SecretCryptor::from_record_keystore(&path, "journal")
        .unwrap()
        .expect("a passphrase seals the journal");
    let created = std::fs::read(&path).expect("the keystore was created");

    open_record_keys(&path)
        .unwrap()
        .expect("the store unseals it");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        created,
        "the store opened the journal's keystore and never wrote a second one"
    );

    let reopened = SecretCryptor::from_record_keystore(&path, "journal")
        .unwrap()
        .unwrap();
    let frame = journal.encrypt_bytes(b"memory frame", b"lsn-7").unwrap();
    assert_eq!(
        reopened.decrypt_bytes(&frame, b"lsn-7").unwrap(),
        b"memory frame"
    );
    std::env::remove_var("VELD_MASTER_PASSPHRASE");
}
