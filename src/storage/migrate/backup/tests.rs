use super::*;
use crate::backup::RestoreOptions;
use crate::graph_memory::{EntityLabel, EntityNode, GraphMemory, PiiClassification};
use crate::memory::storage::VectorMappingEntry;
use crate::memory::types::{Experience, ExperienceType, Memory, MemoryId, Todo};
use crate::memory::TodoStore;
use crate::storage::redb::{RedbAuditStore, RedbColumnStore, RedbPrimaryMemoryStore};
use crate::storage::{AuditLogEntry, AuditStore, PrimaryMemoryStore};
use std::sync::Arc;

const USER: &str = "alice";

fn memory(content: &str) -> Memory {
    Memory::new(
        MemoryId(uuid::Uuid::new_v4()),
        Experience {
            experience_type: ExperienceType::Observation,
            content: content.to_string(),
            ..Default::default()
        },
        0.5,
        None,
        None,
        None,
        None,
    )
}

fn entity(name: &str) -> EntityNode {
    EntityNode {
        uuid: uuid::Uuid::new_v4(),
        name: name.to_string(),
        labels: vec![EntityLabel::Concept],
        created_at: chrono::Utc::now(),
        last_seen_at: chrono::Utc::now(),
        mention_count: 1,
        summary: String::new(),
        attributes: std::collections::HashMap::new(),
        name_embedding: None,
        salience: 0.5,
        is_proper_noun: false,
        pii_classification: PiiClassification::Clean,
    }
}

/// A redb data directory whose stores stay open for the fixture's life, as a server's do.
struct Live {
    primary: RedbPrimaryMemoryStore,
    graph_db: Arc<::redb::Database>,
    graph: GraphMemory,
    shared_db: Arc<::redb::Database>,
    todos: TodoStore,
    audit: RedbAuditStore,
    memories: Vec<Memory>,
    entity: uuid::Uuid,
    todo: Todo,
    journal: Vec<u8>,
}

impl Live {
    fn open(base: &Path) -> Self {
        let user = base.join(USER);
        let primary = RedbPrimaryMemoryStore::open(&user).unwrap();
        let memories: Vec<Memory> = ["one", "two", "three"].iter().map(|c| memory(c)).collect();
        for (i, m) in memories.iter().enumerate() {
            primary.put_memory(m).unwrap();
            let mut mapping = VectorMappingEntry::default();
            mapping.add_modality(
                crate::memory::storage::Modality::Text,
                vec![i as u32, i as u32 + 7],
            );
            primary.put_vector_mapping(&m.id, &mapping).unwrap();
        }
        primary
            .key_value_store()
            .put_bytes(b"dials", b"tuned")
            .unwrap();

        let graph_db = crate::storage::redb::open_graph_database(&user.join("graph")).unwrap();
        let graph =
            GraphMemory::with_store(Arc::new(RedbColumnStore::new(Arc::clone(&graph_db)))).unwrap();
        let entity = graph.add_entity(entity("Rust")).unwrap();

        let shared_db = crate::storage::redb::open_shared_database(base).unwrap();
        let todos =
            TodoStore::with_store(Arc::new(RedbColumnStore::new(Arc::clone(&shared_db))), base)
                .unwrap();
        let todo = todos
            .store_todo(&Todo::new(USER.into(), "back it up".into()))
            .unwrap();
        let audit = RedbAuditStore::open(base).unwrap();
        audit
            .append_event(
                USER,
                &AuditLogEntry {
                    timestamp: chrono::Utc::now(),
                    event_type: "backup-test".into(),
                    memory_id: "m".into(),
                    details: "d".into(),
                },
            )
            .unwrap();

        {
            let mut log = crate::intent_log::IntentLog::open(user.join(JOURNAL)).unwrap();
            log.append(b"first frame").unwrap();
            log.append(b"second frame").unwrap();
            log.sync().unwrap();
        }
        let journal = std::fs::read(user.join(JOURNAL)).unwrap();
        Self {
            primary,
            graph_db,
            graph,
            shared_db,
            todos,
            audit,
            memories,
            entity,
            todo,
            journal,
        }
    }

    fn handles(&self) -> (Arc<::redb::Database>, Arc<::redb::Database>) {
        (self.primary.database(), self.audit.database())
    }
}

fn backup(engine: &RedbBackupEngine, base: &Path, live: &Live) -> BackupMetadata {
    let (primary, audit) = live.handles();
    engine
        .create(
            base,
            USER,
            &LiveRedb {
                primary: Some(&primary),
                graph: Some(&live.graph_db),
                audit: Some(&audit),
                shared: Some(&live.shared_db),
            },
            &IntentLogBackupSpec::per_user(base.join(USER)),
        )
        .expect("create backup")
}

fn setup() -> (tempfile::TempDir, RedbBackupEngine, Live) {
    let tmp = tempfile::tempdir().unwrap();
    let engine = RedbBackupEngine::new(tmp.path().join("backups/.redb")).unwrap();
    let live = Live::open(tmp.path());
    (tmp, engine, live)
}

fn dump_path(engine: &RedbBackupEngine, id: u32, unit: &Unit) -> PathBuf {
    engine
        .backup_dir(USER, id)
        .unwrap()
        .join(EXPORT_DIR)
        .join(dump_file_name(unit))
}

fn user_unit() -> Unit {
    Unit::User(USER.to_string())
}

#[test]
fn a_round_trip_reads_every_record_back_through_the_redb_stores() {
    let (tmp, engine, live) = setup();
    let first = backup(&engine, tmp.path(), &live);
    live.primary.put_memory(&memory("after the first")).unwrap();
    let second = backup(&engine, tmp.path(), &live);

    assert_eq!((first.backup_id, second.backup_id), (1, 2));
    assert_eq!(first.memory_count, 3);
    assert_eq!(second.memory_count, 4);
    assert!(first.intent_log_files.iter().any(|f| f.name == JOURNAL));
    assert!(first.journal_cut_lsn.is_some());
    assert_eq!(
        first.secondary_stores,
        vec![
            "audit".to_string(),
            "graph".to_string(),
            "shared".to_string()
        ]
    );
    let listed: Vec<u32> = engine
        .list(USER)
        .unwrap()
        .iter()
        .map(|m| m.backup_id)
        .collect();
    assert_eq!(listed, vec![2, 1], "newest first");
    for id in [1, 2] {
        engine
            .verify(USER, id)
            .expect("an untouched backup verifies");
    }

    let target = tempfile::tempdir().unwrap();
    let report = engine
        .restore_into(USER, 1, target.path(), &RestoreOptions::default(), true)
        .expect("restore");
    assert!(report.not_restored.is_empty());
    assert_eq!(report.imported.len(), 2);
    for imported in &report.imported {
        assert_eq!(
            (imported.verified_frames, &imported.verified_digest),
            (imported.frames, &imported.digest)
        );
    }

    let restored_user = target.path().join(USER);
    let primary = RedbPrimaryMemoryStore::open(&restored_user).unwrap();
    let mut got: Vec<uuid::Uuid> = primary.get_all().unwrap().iter().map(|m| m.id.0).collect();
    got.sort();
    let mut want: Vec<uuid::Uuid> = live.memories.iter().map(|m| m.id.0).collect();
    want.sort();
    assert_eq!(
        got, want,
        "backup 1 holds exactly the three memories it was taken over"
    );
    for m in &live.memories {
        assert_eq!(
            primary.get(&m.id).unwrap().experience.content,
            m.experience.content
        );
        assert!(primary.get_vector_mapping(&m.id).unwrap().is_some());
    }
    assert_eq!(
        primary.key_value_store().get_bytes(b"dials").unwrap(),
        Some(b"tuned".to_vec())
    );

    let graph_db = crate::storage::redb::open_graph_database(&restored_user.join("graph")).unwrap();
    let graph = GraphMemory::with_store(Arc::new(RedbColumnStore::new(graph_db))).unwrap();
    assert_eq!(
        graph.get_entity(&live.entity).unwrap().map(|e| e.name),
        Some("Rust".to_string())
    );

    let shared_db = crate::storage::redb::open_shared_database(target.path()).unwrap();
    let todos =
        TodoStore::with_store(Arc::new(RedbColumnStore::new(shared_db)), target.path()).unwrap();
    assert_eq!(
        todos
            .get_todo(USER, &live.todo.id)
            .unwrap()
            .map(|t| t.content),
        Some("back it up".to_string())
    );
    let audit = RedbAuditStore::open(target.path()).unwrap();
    assert_eq!(audit.list_events(USER, 10).unwrap().len(), 1);

    assert_eq!(
        std::fs::read(restored_user.join(JOURNAL)).unwrap(),
        live.journal
    );
    drop(live.graph);
    drop(live.todos);
}

#[test]
fn a_restore_without_the_shared_stores_names_them_and_never_merges() {
    let (tmp, engine, live) = setup();
    backup(&engine, tmp.path(), &live);
    let target = tempfile::tempdir().unwrap();
    let report = engine
        .restore_into(USER, 1, target.path(), &RestoreOptions::default(), false)
        .unwrap();
    assert_eq!(report.not_restored.len(), 1);
    assert!(report.not_restored[0].contains("not restored"));
    assert!(!target.path().join("shared.redb").exists());

    let again = engine
        .restore_into(USER, 1, target.path(), &RestoreOptions::default(), false)
        .expect_err("a populated target must be refused");
    assert!(format!("{again:#}").contains("already exists"), "{again:#}");
}

#[test]
fn verify_fails_on_a_tampered_a_missing_and_a_truncated_dump() {
    let (tmp, engine, live) = setup();
    for _ in 0..4 {
        backup(&engine, tmp.path(), &live);
    }

    let tampered = dump_path(&engine, 1, &user_unit());
    let mut bytes = std::fs::read(&tampered).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x55;
    std::fs::write(&tampered, bytes).unwrap();
    let err = format!(
        "{:#}",
        engine.verify(USER, 1).expect_err("tampered dump verified")
    );
    assert!(err.contains("digest mismatch"), "{err}");

    std::fs::remove_file(dump_path(&engine, 2, &Unit::Shared)).unwrap();
    let err = format!(
        "{:#}",
        engine.verify(USER, 2).expect_err("missing dump verified")
    );
    assert!(err.contains("shared.veldport"), "{err}");

    let truncated = dump_path(&engine, 3, &user_unit());
    let len = std::fs::metadata(&truncated).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&truncated)
        .unwrap()
        .set_len(len - 10)
        .unwrap();
    let err = format!(
        "{:#}",
        engine.verify(USER, 3).expect_err("truncated dump verified")
    );
    assert!(err.contains("does not end with a trailer"), "{err}");

    let journal = engine
        .backup_dir(USER, 4)
        .unwrap()
        .join(INTENT_DIR)
        .join(JOURNAL);
    let mut bytes = std::fs::read(&journal).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&journal, bytes).unwrap();
    let err = format!(
        "{:#}",
        engine
            .verify(USER, 4)
            .expect_err("altered intent log verified")
    );
    assert!(err.contains("CRC mismatch"), "{err}");
}

#[test]
fn restore_refuses_a_backup_whose_dump_was_swapped_for_another_valid_one() {
    let (tmp, engine, live) = setup();
    backup(&engine, tmp.path(), &live);
    live.primary
        .put_memory(&memory("only in backup two"))
        .unwrap();
    backup(&engine, tmp.path(), &live);
    std::fs::copy(
        dump_path(&engine, 1, &user_unit()),
        dump_path(&engine, 2, &user_unit()),
    )
    .unwrap();

    let target = tempfile::tempdir().unwrap();
    let err = engine
        .restore_into(USER, 2, target.path(), &RestoreOptions::default(), false)
        .expect_err("a dump that is valid on its own but not this backup's must not restore");
    assert!(!target.path().join(USER).join("storage.redb").exists());
    assert!(format!("{err:#}").contains("backup 2"), "{err:#}");
}

#[test]
fn purge_keeps_exactly_the_n_newest() {
    let (tmp, engine, live) = setup();
    for _ in 0..5 {
        backup(&engine, tmp.path(), &live);
    }
    assert!(engine.purge(USER, 0).is_err());
    assert_eq!(engine.purge(USER, 10).unwrap(), 0);
    assert_eq!(engine.purge(USER, 2).unwrap(), 3);
    let kept: Vec<u32> = engine
        .list(USER)
        .unwrap()
        .iter()
        .map(|m| m.backup_id)
        .collect();
    assert_eq!(kept, vec![5, 4]);
    let next = backup(&engine, tmp.path(), &live);
    assert_eq!(next.backup_id, 6, "a purged id is never reused");
}

#[test]
fn an_unfinished_backup_leaves_nothing_behind() {
    let (tmp, engine, live) = setup();
    let (primary, audit) = live.handles();
    {
        let mut pending = engine.begin(USER).unwrap();
        pending
            .export(
                tmp.path(),
                &LiveRedb {
                    primary: Some(&primary),
                    graph: Some(&live.graph_db),
                    audit: Some(&audit),
                    shared: Some(&live.shared_db),
                },
            )
            .unwrap();
        let err = pending.finish().expect_err("no intent log was staged");
        assert!(format!("{err:#}").contains("intent log"), "{err:#}");
    }
    assert!(engine.list(USER).unwrap().is_empty());
    let leftovers: Vec<_> = std::fs::read_dir(engine.user_root(USER).unwrap())
        .unwrap()
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn a_file_swap_refuses_an_existing_aside_and_otherwise_replaces_all() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("a"), b"old a").unwrap();
    std::fs::write(dir.join("b"), b"old b").unwrap();
    std::fs::write(dir.join("a.new"), b"new a").unwrap();
    std::fs::write(dir.join("b.pre_restore"), b"survivor").unwrap();
    let pairs = vec![
        (dir.join("a.new"), dir.join("a")),
        (dir.join("b.absent"), dir.join("b")),
    ];
    assert!(swap_files_into_place(&pairs).is_err());
    assert_eq!(std::fs::read(dir.join("a")).unwrap(), b"old a");
    assert_eq!(
        std::fs::read(dir.join("b.pre_restore")).unwrap(),
        b"survivor"
    );

    std::fs::remove_file(dir.join("b.pre_restore")).unwrap();
    swap_files_into_place(&pairs).unwrap();
    assert_eq!(std::fs::read(dir.join("a")).unwrap(), b"new a");
    assert!(
        !dir.join("b").exists(),
        "a target with nothing staged is emptied"
    );
    assert!(!dir.join("a.pre_restore").exists() && !dir.join("b.pre_restore").exists());
}
