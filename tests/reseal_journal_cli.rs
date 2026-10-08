//! `veld reseal-journal` seals what a passphrase-only tenant wrote in plaintext before
//! its journal and slow store took their key from the store keystore: every journal
//! frame and every slow-store payload, under the key the server now uses for them.

use std::process::Command;
use tempfile::TempDir;
use veld::intent_log::IntentLog;
use veld::keystore::{open_record_keys, SecretCryptor};
use veld::memory::slow_store::SlowStore;

const PASSPHRASE: &str = "reseal-cli-correct-horse-battery-staple-R5";
const SECRET: &[u8] = b"pre-fix plaintext memory snapshot R5";

fn holds_secret(path: &std::path::Path) -> bool {
    std::fs::read(path)
        .map(|b| b.windows(SECRET.len()).any(|w| w == SECRET))
        .unwrap_or(false)
}

fn reseal(data: &std::path::Path, extra: &[&str]) -> serde_json::Value {
    let out = Command::new(env!("CARGO_BIN_EXE_veld"))
        .args(["reseal-journal", "--backend", "rocksdb", "--storage"])
        .arg(data)
        .args(extra)
        .env("VELD_MASTER_PASSPHRASE", PASSPHRASE)
        .env_remove("VELD_KEYSTORE_PATH")
        .output()
        .expect("run veld");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("JSON report")
}

#[test]
fn the_one_off_seals_the_journal_and_slow_store_a_passphrase_tenant_wrote_in_plaintext() {
    std::env::remove_var("VELD_KEYSTORE_PATH");
    std::env::remove_var("VELD_MASTER_PASSPHRASE");
    let data = TempDir::new().unwrap();
    let tenant = data.path().join("alice");
    std::fs::create_dir_all(tenant.join("storage")).unwrap();
    {
        let mut log = IntentLog::open(tenant.join("intent.log")).unwrap();
        log.append(SECRET).unwrap();
        log.sync().unwrap();
        let slow = SlowStore::open(&tenant.join("slow_store.db")).unwrap();
        slow.upsert_memory("alice", "m-1", 0, SECRET, 0.5).unwrap();
    }
    std::env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);
    open_record_keys(&tenant.join("storage").join("keystore.json"))
        .unwrap()
        .expect("the store keystore the server would have created");
    std::env::remove_var("VELD_MASTER_PASSPHRASE");
    assert!(holds_secret(&tenant.join("intent.log")));

    let dry = reseal(data.path(), &["--dry-run"]);
    assert_eq!(dry[0]["resealed"], 1);
    assert_eq!(dry[0]["slow_store"]["plaintext_rows"], 1);
    assert!(
        holds_secret(&tenant.join("intent.log")),
        "a dry run writes nothing"
    );

    let run = reseal(data.path(), &[]);
    assert_eq!(run[0]["rewritten"], true);
    assert_eq!(run[0]["slow_store"]["sealed"], true);
    assert!(
        !holds_secret(&tenant.join("intent.log")),
        "the journal is sealed"
    );
    for file in ["slow_store.db", "slow_store.db-wal"] {
        assert!(!holds_secret(&tenant.join(file)), "{file} is sealed");
    }

    std::env::set_var("VELD_MASTER_PASSPHRASE", PASSPHRASE);
    let key = SecretCryptor::from_record_keystore(
        &tenant.join("storage").join("keystore.json"),
        "journal",
    )
    .unwrap()
    .unwrap();
    let log = IntentLog::open_with_cryptor(tenant.join("intent.log"), Some(key)).unwrap();
    let frames: Vec<_> = log.iter().unwrap().map(|r| r.unwrap()).collect();
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].payload, SECRET,
        "the frame opens under the server's journal key"
    );
    std::env::remove_var("VELD_MASTER_PASSPHRASE");

    let again = reseal(data.path(), &[]);
    assert_eq!(
        (
            again[0]["resealed"].clone(),
            again[0]["slow_store"]["plaintext_rows"].clone()
        ),
        (0.into(), 0.into())
    );
}
