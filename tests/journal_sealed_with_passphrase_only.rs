//! VELD_MASTER_PASSPHRASE alone seals a tenant's records. Its journal carries the same
//! memories as full snapshots, so it is sealed too, and the BM25 projection over it
//! stays in memory: neither is a plaintext copy beside the sealed store.

use tempfile::TempDir;
use veld::config::ServerConfig;
use veld::handlers::MultiUserMemoryManager;
use veld::intent_log::payload::CURRENT_PAYLOAD_SCHEMA_VERSION;
use veld::intent_log::IntentPayload;
use veld::memory::{Experience, ExperienceType, Memory, MemoryId};

const SECRET: &str = "journal-sealed-distinctive-plaintext-do-not-leak-P2";

#[test]
fn a_passphrase_alone_seals_the_journal_and_keeps_bm25_off_disk() {
    std::env::remove_var("VELD_KEYSTORE_PATH");
    std::env::set_var(
        "VELD_MASTER_PASSPHRASE",
        "journal-sealed-correct-horse-battery-P2",
    );
    std::env::set_var("VELD_LAZY_CROSS_ENCODER", "1");

    let dir = TempDir::new().unwrap();
    let cfg = ServerConfig {
        storage_path: dir.path().to_path_buf(),
        backup_enabled: false,
        ..ServerConfig::default()
    };
    let mgr = MultiUserMemoryManager::new(dir.path().to_path_buf(), cfg).expect("manager");

    let id = MemoryId(uuid::Uuid::new_v4());
    let memory = Memory::new(
        id.clone(),
        Experience {
            experience_type: ExperienceType::Observation,
            content: SECRET.to_string(),
            ..Default::default()
        },
        0.5,
        None,
        None,
        None,
        None,
    );
    let snapshot = bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
    mgr.journal_and_apply(
        "alice",
        &IntentPayload::Remember {
            user_id: "alice".into(),
            memory_id: id.0.to_string(),
            memory_bincode: snapshot,
            schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
        },
    )
    .expect("journal");

    let tenant = dir.path().join("alice");
    let raw = std::fs::read(tenant.join("intent.log")).expect("journal on disk");
    assert!(
        !raw.windows(SECRET.len()).any(|w| w == SECRET.as_bytes()),
        "the journal holds the memory in plaintext"
    );
    for file in ["slow_store.db", "slow_store.db-wal"] {
        if let Ok(bytes) = std::fs::read(tenant.join(file)) {
            assert!(
                !bytes.windows(SECRET.len()).any(|w| w == SECRET.as_bytes()),
                "{file} holds the memory in plaintext"
            );
        }
    }
    assert!(
        !tenant.join(veld::memory::BM25_INDEX_DIR).exists(),
        "the BM25 projection was written to disk"
    );
    std::env::remove_var("VELD_MASTER_PASSPHRASE");
}
