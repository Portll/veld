use super::*;
use crate::graph_memory::{EntityLabel, EntityNode, GraphMemory, PiiClassification};
use crate::keystore::{Keystore, RecordCryptors};
#[cfg(feature = "storage-redb")]
use crate::memory::storage::SearchCriteria;
use crate::memory::storage::VectorMappingEntry;
use crate::memory::types::{Experience, ExperienceType, Memory, MemoryId, MemoryTier};
use crate::storage::legacy_rocksdb::RocksDbPrimaryMemoryStore;
#[cfg(feature = "storage-redb")]
use crate::storage::redb::RedbPrimaryMemoryStore;
use crate::storage::PrimaryMemoryStore;
#[cfg(feature = "storage-redb")]
use sha2::{Digest, Sha256};

fn memory(content: &str, tag: &str) -> Memory {
    let now = chrono::Utc::now();
    Memory::from_legacy(
        MemoryId(uuid::Uuid::new_v4()),
        Experience {
            experience_type: ExperienceType::Observation,
            content: content.to_string(),
            tags: vec![tag.to_string()],
            ..Default::default()
        },
        0.5,
        0,
        now,
        now,
        false,
        MemoryTier::LongTerm,
        Vec::new(),
        1.0,
        None,
        None,
        None,
        None,
        0.0,
        None,
        None,
        1,
        Vec::new(),
        Vec::new(),
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

fn sorted_ids(memories: &[Memory]) -> Vec<uuid::Uuid> {
    let mut ids: Vec<_> = memories.iter().map(|m| m.id.0).collect();
    ids.sort();
    ids
}

fn mapping(id: u32) -> VectorMappingEntry {
    let mut entry = VectorMappingEntry::default();
    entry.add_modality(crate::memory::storage::Modality::Text, vec![id, id + 1]);
    entry
}

/// A rocksdb user with memories, mappings, a watermark, a kv row and one graph entity.
/// Primary records carried: 2 per memory + 2.
fn rocks_user(base: &Path, id: &str, tags: &[&str]) -> (Vec<Memory>, uuid::Uuid) {
    let dir = base.join(id);
    let store = RocksDbPrimaryMemoryStore::open(&dir, None).unwrap();
    let memories: Vec<Memory> = tags
        .iter()
        .map(|t| memory(&format!("about {t}"), t))
        .collect();
    for (i, m) in memories.iter().enumerate() {
        store.put_memory(m).unwrap();
        store
            .put_vector_mapping(&m.id, &mapping(i as u32 * 10))
            .unwrap();
    }
    store.set_fact_watermark(id, 1_726_000_000_000);
    store
        .key_value_store()
        .put_bytes(b"dials", b"tuned")
        .unwrap();
    store.flush().unwrap();
    drop(store);
    let graph = GraphMemory::new(&dir.join("graph"), None).unwrap();
    let uuid = graph.add_entity(entity("Rust")).unwrap();
    drop(graph);
    (memories, uuid)
}

#[cfg(feature = "storage-redb")]
fn rocks_rows(dir: &Path) -> u64 {
    let store = readonly::open_rocks(dir).unwrap().expect("store present");
    let mut n = 0;
    for cf in &store.column_families {
        let handle = store.db.cf_handle(cf).unwrap();
        n += store
            .db
            .iterator_cf(handle, rocksdb::IteratorMode::Start)
            .count() as u64;
    }
    n
}

/// Every file under `root` with a content hash, so "unchanged" is measured, not assumed.
#[cfg(feature = "storage-redb")]
fn snapshot(root: &Path) -> BTreeMap<PathBuf, String> {
    fn walk(dir: &Path, root: &Path, out: &mut BTreeMap<PathBuf, String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, root, out);
            } else {
                let bytes = std::fs::read(&path).unwrap();
                let digest: [u8; 32] = Sha256::digest(&bytes).into();
                out.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    portable::hex(&digest),
                );
            }
        }
    }
    let mut out = BTreeMap::new();
    if root.exists() {
        walk(root, root, &mut out);
    }
    out
}

#[cfg(feature = "storage-redb")]
fn plant_undecodable_memory(base: &Path, id: &str) -> uuid::Uuid {
    plant_undecodable_rocks_memory(&base.join(id))
}

#[cfg(feature = "storage-redb")]
fn logical_dump(path: &Path, frames: &[(&str, &[u8], &[u8])]) {
    let mut buf = Vec::new();
    let mut w = PortableWriter::new(&mut buf).unwrap();
    for (cf, key, value) in frames {
        w.write_record(&PortableRecord {
            cf: cf.to_string(),
            key: key.to_vec(),
            value: value.to_vec(),
        })
        .unwrap();
    }
    w.finish().unwrap();
    std::fs::write(path, buf).unwrap();
}

fn user(id: &str) -> Scope {
    Scope::User(id.to_string())
}

#[cfg(feature = "storage-redb")]
#[test]
fn rocks_to_redb_is_read_back_by_the_redb_stores_themselves() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    let (memories, entity_id) = rocks_user(base, "alice", &["rust", "redb", "cutover"]);

    let report = migrate(
        base,
        StorageBackend::RocksDb,
        StorageBackend::Redb,
        &user("alice"),
        false,
        None,
    )
    .expect("migrate");
    assert_eq!(report.imported.len(), 1);
    let imported = &report.imported[0];
    assert_eq!(
        (imported.verified_frames, &imported.verified_digest),
        (imported.frames, &imported.digest)
    );

    let redb = RedbPrimaryMemoryStore::open(&base.join("alice")).unwrap();
    assert_eq!(sorted_ids(&redb.get_all().unwrap()), sorted_ids(&memories));
    for m in &memories {
        assert_eq!(
            redb.get(&m.id).unwrap().experience.content,
            m.experience.content
        );
    }
    let got = redb
        .get_vector_mapping(&memories[1].id)
        .unwrap()
        .expect("mapping carried");
    assert_eq!(got.modalities.len(), 1);
    assert_eq!(redb.get_fact_watermark("alice"), Some(1_726_000_000_000));
    assert_eq!(
        redb.key_value_store().get_bytes(b"dials").unwrap(),
        Some(b"tuned".to_vec())
    );
    drop(redb);

    let graph_db = crate::storage::redb::open_graph_database(&base.join("alice/graph")).unwrap();
    let graph = GraphMemory::with_store(std::sync::Arc::new(
        crate::storage::redb::RedbColumnStore::new(graph_db),
    ))
    .unwrap();
    assert_eq!(
        graph.get_entity(&entity_id).unwrap().map(|e| e.name),
        Some("Rust".to_string())
    );
}

#[cfg(feature = "storage-redb")]
#[test]
fn redb_to_rocks_rebuilds_the_index_rocksdb_search_depends_on() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    let dir = base.join("bob");
    let tagged = memory("index me", "findable");
    {
        let redb = RedbPrimaryMemoryStore::open(&dir).unwrap();
        redb.put_memory(&tagged).unwrap();
        redb.put_memory(&memory("other", "elsewhere")).unwrap();
    }

    migrate(
        base,
        StorageBackend::Redb,
        StorageBackend::RocksDb,
        &user("bob"),
        false,
        None,
    )
    .expect("migrate");

    let rocks = RocksDbPrimaryMemoryStore::open(&dir, None).unwrap();
    assert_eq!(rocks.get_all().unwrap().len(), 2);
    let found = rocks
        .search(SearchCriteria::ByTags(vec!["findable".into()]))
        .unwrap();
    assert_eq!(
        sorted_ids(&found),
        vec![tagged.id.0],
        "memory_index was not rebuilt on the rocksdb target"
    );
}

#[cfg(feature = "storage-redb")]
#[test]
fn shared_stores_cross_backends_with_audit_in_its_own_file() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    {
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let db = rocksdb::DB::open_cf(&opts, base.join("shared"), ["default", "audit", "todos"])
            .unwrap();
        db.put_cf(
            db.cf_handle("audit").unwrap(),
            b"alice:00000000000000000001",
            b"event",
        )
        .unwrap();
        db.put_cf(
            db.cf_handle("todos").unwrap(),
            b"todo-1",
            b"write the migration",
        )
        .unwrap();
        db.flush().unwrap();
    }

    let report = migrate(
        base,
        StorageBackend::RocksDb,
        StorageBackend::Redb,
        &Scope::All,
        false,
        None,
    )
    .unwrap();
    assert_eq!(
        report
            .imported
            .iter()
            .map(|r| r.unit.clone())
            .collect::<Vec<_>>(),
        vec![Unit::Shared]
    );

    let read_row = |file: &str, table: &str, key: &[u8]| -> Option<Vec<u8>> {
        let db = redb::Database::open(base.join(file)).unwrap();
        let read = db.begin_read().unwrap();
        let t = read.open_table(RawTable::new(table)).unwrap();
        let v = redb::ReadableTable::get(&t, key)
            .unwrap()
            .map(|v| v.value().to_vec());
        v
    };
    assert_eq!(
        read_row("audit.redb", "audit", b"alice:00000000000000000001"),
        Some(b"event".to_vec())
    );
    assert_eq!(
        read_row("shared.redb", "todos", b"todo-1"),
        Some(b"write the migration".to_vec())
    );
}

#[cfg(feature = "storage-redb")]
#[test]
fn an_undecodable_record_fails_the_export_and_is_counted_not_skipped() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "carol", &["a", "b"]);
    let bad = plant_undecodable_memory(base, "carol");

    let out = tmp.path().join("export");
    let err = export(base, StorageBackend::RocksDb, &user("carol"), &out, None)
        .expect_err("an export with an undecodable record must fail");
    let msg = format!("{err:#}");
    assert!(msg.contains("1 record(s) failed to decode"), "{msg}");
    assert!(
        msg.contains(&bad.to_string()),
        "the failing key must be named: {msg}"
    );
    assert_eq!(
        snapshot(&out),
        BTreeMap::new(),
        "a failed export left files behind"
    );

    let plan = migrate(
        base,
        StorageBackend::RocksDb,
        StorageBackend::Redb,
        &user("carol"),
        true,
        None,
    )
    .unwrap();
    assert!(plan.blocked());
    let source = plan.units[0].source.as_ref().unwrap();
    assert_eq!(source.decode_failures, 1);
    assert_eq!(
        source.records,
        2 * 2 + 2 + rocks_rows(&base.join("carol/graph/graph"))
    );
    let err = migrate(
        base,
        StorageBackend::RocksDb,
        StorageBackend::Redb,
        &user("carol"),
        false,
        None,
    )
    .expect_err("a real migration must refuse");
    assert!(format!("{err:#}").contains("refused before writing anything"));
    assert!(
        !base.join("carol/storage.redb").exists(),
        "a refused migration created the target"
    );
}

#[test]
fn a_sixteen_byte_row_under_a_subsystem_prefix_is_not_mistaken_for_a_memory() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "cory", &["a"]);
    {
        let db = rocksdb::DB::open_cf(
            &rocksdb::Options::default(),
            base.join("cory/storage"),
            ["default", "kv", "memory_index"],
        )
        .unwrap();
        db.put(b"facts:0123456789", b"not a memory").unwrap();
        db.flush().unwrap();
    }
    let seen = extract(
        base,
        StorageBackend::RocksDb,
        &Unit::User("cory".into()),
        std::io::sink(),
    )
    .unwrap();
    assert_eq!(seen.decode_failures, 0, "{:?}", seen.failed_keys);
}

#[cfg(feature = "storage-redb")]
#[test]
fn an_import_whose_rows_do_not_land_as_declared_fails_verification() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    let dump = tmp.path().join("dup.veldport");
    // Two frames for one key: the codec accepts it, the store keeps one row.
    logical_dump(
        &dump,
        &[
            (META, b"format", FORMAT.as_bytes()),
            (META, b"unit", b"user"),
            (META, b"user_id", b"dave"),
            ("primary/kv", b"k", b"first"),
            ("primary/kv", b"k", b"second"),
        ],
    );
    let err =
        import(base, StorageBackend::Redb, &dump).expect_err("a mismatch must not report success");
    let msg = format!("{err:#}");
    assert!(msg.contains("verification FAILED"), "{msg}");
    assert!(msg.contains("declares 5 frames"), "{msg}");
    assert!(msg.contains("re-derives 4 frames"), "{msg}");
}

#[cfg(feature = "storage-redb")]
#[test]
fn verify_catches_a_target_changed_after_a_good_import() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "erin", &["x"]);
    let out = tmp.path().join("export");
    let dumps = export(base, StorageBackend::RocksDb, &user("erin"), &out, None).unwrap();
    let target = tmp.path().join("target");
    std::fs::create_dir_all(&target).unwrap();
    import(&target, StorageBackend::Redb, &out).expect("import");

    let (frames, digest) = portable::read_trailer(&dumps[0].path).unwrap();
    let unit = Unit::User("erin".into());
    verify(&target, StorageBackend::Redb, &unit, frames, &digest).expect("faithful");
    RedbPrimaryMemoryStore::open(&target.join("erin"))
        .unwrap()
        .key_value_store()
        .put_bytes(b"stowaway", b"1")
        .unwrap();
    let err = verify(&target, StorageBackend::Redb, &unit, frames, &digest)
        .expect_err("an extra row must fail verification");
    assert!(format!("{err:#}").contains("verification FAILED"));
}

#[cfg(feature = "storage-redb")]
#[test]
fn dry_run_writes_nothing_on_either_side() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    let (memories, _) = rocks_user(base, "frank", &["p", "q", "r"]);
    // An empty redb file already present, as on the live box.
    drop(RedbPrimaryMemoryStore::open(&base.join("frank")).unwrap());
    let before = snapshot(base);

    let plan = migrate(
        base,
        StorageBackend::RocksDb,
        StorageBackend::Redb,
        &user("frank"),
        true,
        None,
    )
    .unwrap();
    assert!(!plan.blocked(), "{:?}", plan.units[0].blocked);
    let source = plan.units[0].source.as_ref().unwrap();
    let graph_rows = rocks_rows(&base.join("frank/graph/graph"));
    assert_eq!(source.records, memories.len() as u64 * 2 + 2 + graph_rows);
    assert!(source.derived_rows_not_carried > 0);
    assert_eq!(plan.units[0].target.as_ref().unwrap().records, 0);
    assert!(plan.imported.is_empty());

    assert_eq!(
        snapshot(base),
        before,
        "a dry run changed the data directory"
    );
}

#[cfg(feature = "storage-redb")]
#[test]
fn a_dry_run_never_creates_an_absent_target() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "gina", &["only"]);
    let before = snapshot(base);
    migrate(
        base,
        StorageBackend::RocksDb,
        StorageBackend::Redb,
        &Scope::All,
        true,
        None,
    )
    .unwrap();
    assert_eq!(snapshot(base), before);
    assert!(!base.join("gina/storage.redb").exists());
}

#[cfg(feature = "storage-redb")]
#[test]
fn migrate_leaves_the_source_byte_identical() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "hank", &["one", "two"]);
    let source_before = snapshot(&base.join("hank/storage"));
    let graph_before = snapshot(&base.join("hank/graph/graph"));

    migrate(
        base,
        StorageBackend::RocksDb,
        StorageBackend::Redb,
        &user("hank"),
        false,
        None,
    )
    .unwrap();

    assert_eq!(
        snapshot(&base.join("hank/storage")),
        source_before,
        "the rocksdb source was modified"
    );
    assert_eq!(
        snapshot(&base.join("hank/graph/graph")),
        graph_before,
        "the rocksdb graph source was modified"
    );
}

#[cfg(feature = "storage-redb")]
#[test]
fn a_redb_source_is_read_without_touching_its_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("ivy");
    RedbPrimaryMemoryStore::open(&dir)
        .unwrap()
        .put_memory(&memory("m", "t"))
        .unwrap();
    let before = snapshot(tmp.path());
    let seen = extract(
        tmp.path(),
        StorageBackend::Redb,
        &Unit::User("ivy".into()),
        std::io::sink(),
    )
    .unwrap();
    assert_eq!(seen.records, 1);
    assert_eq!(
        snapshot(tmp.path()),
        before,
        "reading a redb store wrote to it"
    );
}

#[cfg(feature = "storage-redb")]
#[test]
fn a_redb_store_held_by_a_writer_is_refused_not_read_mid_write() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("jack");
    let live = RedbPrimaryMemoryStore::open(&dir).unwrap();
    live.put_memory(&memory("m", "t")).unwrap();
    let err = extract(
        tmp.path(),
        StorageBackend::Redb,
        &Unit::User("jack".into()),
        std::io::sink(),
    )
    .expect_err("a store held open for writing must refuse");
    assert!(
        format!("{err:#}").contains("held open for writing"),
        "{err:#}"
    );
    drop(live);
}

#[cfg(feature = "storage-redb")]
#[test]
fn a_populated_target_refuses_the_whole_migration_before_writing() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "kate", &["a"]);
    rocks_user(base, "liam", &["b"]);
    RedbPrimaryMemoryStore::open(&base.join("liam"))
        .unwrap()
        .put_memory(&memory("already", "here"))
        .unwrap();
    let before = snapshot(base);

    let err = migrate(
        base,
        StorageBackend::RocksDb,
        StorageBackend::Redb,
        &Scope::All,
        false,
        None,
    )
    .expect_err("a populated target must refuse");
    assert!(format!("{err:#}").contains("liam"), "{err:#}");
    assert_eq!(
        snapshot(base),
        before,
        "kate was migrated even though liam refused"
    );
}

#[cfg(feature = "storage-redb")]
#[test]
fn an_export_directory_missing_a_listed_dump_does_not_import() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("data");
    rocks_user(&base, "mia", &["a"]);
    rocks_user(&base, "noah", &["b"]);
    let out = tmp.path().join("export");
    let dumps = export(&base, StorageBackend::RocksDb, &Scope::All, &out, None).unwrap();
    assert_eq!(dumps.len(), 2);
    std::fs::remove_file(out.join("users/noah.veldport")).unwrap();

    let target = tmp.path().join("target");
    std::fs::create_dir_all(&target).unwrap();
    assert!(import(&target, StorageBackend::Redb, &out).is_err());
    assert_eq!(snapshot(&target), BTreeMap::new());
}

#[test]
fn export_then_import_round_trips_into_another_data_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("a");
    let (memories, _) = rocks_user(&base, "olga", &["x", "y"]);
    let out = tmp.path().join("export");
    export(&base, StorageBackend::RocksDb, &Scope::All, &out, None).unwrap();

    let target = tmp.path().join("b");
    std::fs::create_dir_all(&target).unwrap();
    let reports = import(&target, StorageBackend::RocksDb, &out).unwrap();
    assert_eq!(reports.len(), 1);
    let rocks = RocksDbPrimaryMemoryStore::open(&target.join("olga"), None).unwrap();
    assert_eq!(sorted_ids(&rocks.get_all().unwrap()), sorted_ids(&memories));
}

#[test]
fn an_encrypted_rocks_store_is_refused_rather_than_decrypted_to_disk() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "pia", &["secret"]);
    std::fs::write(base.join("pia/storage/keystore.json"), b"{}").unwrap();
    let err = export(
        base,
        StorageBackend::RocksDb,
        &user("pia"),
        &tmp.path().join("out"),
        None,
    )
    .expect_err("encrypted stores must refuse");
    assert!(
        format!("{err:#}").contains("record-level encryption"),
        "{err:#}"
    );
}

#[test]
fn a_named_user_with_no_store_is_an_error_not_an_empty_dump() {
    let tmp = tempfile::tempdir().unwrap();
    let err = export(
        tmp.path(),
        StorageBackend::RocksDb,
        &user("nobody"),
        &tmp.path().join("out"),
        None,
    )
    .expect_err("absent user");
    assert!(format!("{err:#}").contains("no rocksdb store for user 'nobody'"));
}

#[cfg(feature = "storage-redb")]
#[test]
fn quarantine_sets_undecodable_records_aside_byte_for_byte_and_accounts_for_them() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("data");
    let (memories, _) = rocks_user(&base, "quinn", &["kept", "also kept"]);
    let bad = plant_undecodable_memory(&base, "quinn");
    let source = extract(
        &base,
        StorageBackend::RocksDb,
        &Unit::User("quinn".into()),
        std::io::sink(),
    )
    .unwrap();
    assert_eq!(source.decode_failures, 1);

    let out = tmp.path().join("export");
    let qdir = tmp.path().join("quarantine");
    let reports = export(
        &base,
        StorageBackend::RocksDb,
        &user("quinn"),
        &out,
        Some(&qdir),
    )
    .expect("quarantined export");
    let r = &reports[0];
    assert_eq!(r.quarantined, 1);
    assert_eq!(r.records + r.quarantined, source.source_records());
    assert_eq!(r.records, source.records);
    let q = r.quarantine.as_ref().expect("quarantine reported");
    assert_eq!(q.ids, vec![format!("primary/memories {bad}")]);

    let (trailer_frames, _) = portable::read_trailer(&r.path).unwrap();
    assert_eq!(trailer_frames, r.frames);
    let exported = portable::read_all(std::fs::File::open(&r.path).unwrap()).unwrap();
    assert!(
        exported.iter().all(|rec| rec.key != bad.as_bytes()),
        "a quarantined record was exported"
    );

    let held = portable::read_all(std::fs::File::open(&q.dump).unwrap()).unwrap();
    assert_eq!(held.len(), 1);
    assert_eq!(
        (
            held[0].cf.as_str(),
            held[0].key.as_slice(),
            held[0].value.as_slice()
        ),
        ("primary/memories", &bad.as_bytes()[..], UNDECODABLE_VALUE)
    );
    let listing: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&q.listing).unwrap()).unwrap();
    let digest: [u8; 32] = Sha256::digest(UNDECODABLE_VALUE).into();
    assert_eq!(
        listing["entries"][0]["value_sha256"],
        portable::hex(&digest)
    );

    let target = tmp.path().join("target");
    std::fs::create_dir_all(&target).unwrap();
    let imported =
        import(&target, StorageBackend::Redb, &out).expect("the shortened dump still verifies");
    assert_eq!(imported[0].verified_frames, r.frames);
    let redb = RedbPrimaryMemoryStore::open(&target.join("quinn")).unwrap();
    assert_eq!(sorted_ids(&redb.get_all().unwrap()), sorted_ids(&memories));
}

#[cfg(feature = "storage-redb")]
#[test]
fn migrate_with_quarantine_migrates_the_rest_and_reports_what_it_set_aside() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("data");
    let (memories, _) = rocks_user(&base, "rhea", &["a", "b", "c"]);
    let bad = plant_undecodable_memory(&base, "rhea");
    let source_before = snapshot(&base.join("rhea/storage"));
    let qdir = tmp.path().join("quarantine");

    let plan = migrate(
        &base,
        StorageBackend::RocksDb,
        StorageBackend::Redb,
        &user("rhea"),
        true,
        Some(&qdir),
    )
    .unwrap();
    assert!(!plan.blocked(), "{:?}", plan.units[0].blocked);
    assert!(!qdir.exists(), "a dry run wrote a quarantine directory");

    let report = migrate(
        &base,
        StorageBackend::RocksDb,
        StorageBackend::Redb,
        &user("rhea"),
        false,
        Some(&qdir),
    )
    .expect("migrate with quarantine");
    assert_eq!(report.quarantined.len(), 1);
    assert_eq!(
        report.quarantined[0].ids,
        vec![format!("primary/memories {bad}")]
    );
    assert_eq!(
        report.imported[0].records + report.quarantined[0].records,
        plan.units[0].source.as_ref().unwrap().source_records()
    );
    let redb = RedbPrimaryMemoryStore::open(&base.join("rhea")).unwrap();
    assert_eq!(sorted_ids(&redb.get_all().unwrap()), sorted_ids(&memories));
    assert_eq!(
        snapshot(&base.join("rhea/storage")),
        source_before,
        "the source was modified"
    );
}

#[test]
fn the_planted_record_is_undecodable_to_the_runtime_reader_and_the_portable_dump_alike() {
    // Guards the plant: these bytes once "decoded" through a raw-UTF-8 arm that fabricated a
    // memory from them (removed 2026-09-16); both readers must now refuse them, so the
    // quarantine tests above exercise a record no arm accepts.
    assert!(crate::memory::storage::portable_memory_value(UNDECODABLE_VALUE).is_err());
    assert!(crate::memory::storage::deserialize_memory_for_test(UNDECODABLE_VALUE).is_err());
}

const SEALED_SECRET: &str = "sealed-in-the-source-and-never-in-a-dump-K4R";

type Row = (Vec<u8>, Vec<u8>);

/// A keystore at `<u>/storage/keystore.json` that has rotated once (epochs 0 and 1,
/// generation 1), with one memory sealed under each epoch and the rollback sentinel at 1.
/// Returns the keystore and the sealed rows as stored.
fn seal_rocks_user(base: &Path, id: &str) -> (Keystore, Vec<Row>) {
    use crate::keystore::KdfParams;
    let storage = base.join(id).join("storage");
    let mut ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
    let kek = ks.unseal_with_passphrase("pw").unwrap();
    let before = RecordCryptors::from_keystore(&ks, &kek).unwrap();
    ks.rotate_dek(&kek).unwrap();
    let after = RecordCryptors::from_keystore(&ks, &kek).unwrap();
    std::fs::write(storage.join(KEYSTORE), ks.to_json().unwrap()).unwrap();

    let db = rocksdb::DB::open_cf(
        &rocksdb::Options::default(),
        &storage,
        ["default", "kv", "memory_index"],
    )
    .unwrap();
    let mut rows = Vec::new();
    for (cryptors, n) in [(&before, 0), (&after, 1)] {
        let m = memory(&format!("{SEALED_SECRET} {n}"), "sealed");
        let plain = bincode::serde::encode_to_vec(&m, bincode::config::standard()).unwrap();
        let sealed = cryptors
            .active()
            .encrypt_record(&plain, m.id.0.as_bytes())
            .unwrap();
        db.put(m.id.0.as_bytes(), &sealed).unwrap();
        rows.push((m.id.0.as_bytes().to_vec(), sealed));
    }
    let index = db.cf_handle("memory_index").unwrap();
    db.put_cf(index, GENERATION_SENTINEL, 1u64.to_le_bytes())
        .unwrap();
    db.flush().unwrap();
    (ks, rows)
}

fn record_keys(ks: &Keystore) -> RecordCryptors {
    let kek = ks.unseal_with_passphrase("pw").unwrap();
    RecordCryptors::from_keystore(ks, &kek).unwrap()
}

fn contains(haystack: &[u8], needle: &str) -> bool {
    haystack
        .windows(needle.len())
        .any(|w| w == needle.as_bytes())
}

#[test]
fn a_migration_dump_carries_sealed_records_as_stored_and_no_plaintext() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "sol", &["plain"]);
    let (ks, sealed) = seal_rocks_user(base, "sol");
    let cryptors = record_keys(&ks);
    let keys = Keys {
        cryptors: &cryptors,
        generation: Some(ks.generation),
    };

    let dump = tmp.path().join("sol.veldport");
    let (stats, _) = export_unit(
        base,
        StorageBackend::RocksDb,
        &Unit::User("sol".to_string()),
        &dump,
        None,
        Sealed::Opened(Some(keys)),
    )
    .expect("a migration export of an encrypted store");
    assert_eq!(stats.sealed_records, 2);
    assert_eq!(stats.decode_failures, 0);

    let bytes = std::fs::read(&dump).unwrap();
    assert!(
        !contains(&bytes, SEALED_SECRET),
        "a sealed record reached the dump opened"
    );
    let records = portable::read_all(bytes.as_slice()).unwrap();
    for (key, value) in &sealed {
        let carried = records
            .iter()
            .find(|r| r.cf == "primary/memories" && &r.key == key)
            .expect("sealed record carried");
        assert_eq!(
            &carried.value, value,
            "a sealed record was not carried as stored"
        );
    }
    let sentinel: Vec<_> = records
        .iter()
        .filter(|r| r.key == GENERATION_SENTINEL)
        .collect();
    assert_eq!(sentinel.len(), 1);
    assert_eq!(sentinel[0].cf, "primary/default");
    assert_eq!(sentinel[0].value, 1u64.to_le_bytes());
    let defaults: Vec<&[u8]> = records
        .iter()
        .filter(|r| r.cf == "primary/default")
        .map(|r| r.key.as_slice())
        .collect();
    assert!(
        defaults.windows(2).all(|w| w[0] < w[1]),
        "the sentinel is out of key order"
    );
}

#[test]
fn a_sealed_record_that_does_not_open_under_the_keystore_is_not_carried() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "tor", &["plain"]);
    let (ks, sealed) = seal_rocks_user(base, "tor");
    let other = Keystore::create("pw", crate::keystore::KdfParams::fast_for_tests()).unwrap();
    let foreign = record_keys(&other)
        .active()
        .encrypt_record(b"never a memory", uuid::Uuid::nil().as_bytes())
        .unwrap();
    let relocated = uuid::Uuid::new_v4();
    {
        let db = rocksdb::DB::open_cf(
            &rocksdb::Options::default(),
            base.join("tor/storage"),
            ["default", "kv", "memory_index"],
        )
        .unwrap();
        db.put(uuid::Uuid::new_v4().as_bytes(), &foreign).unwrap();
        db.put(relocated.as_bytes(), &sealed[0].1).unwrap();
        db.flush().unwrap();
    }
    let cryptors = record_keys(&ks);
    let extraction = extract_with(
        base,
        StorageBackend::RocksDb,
        &Unit::User("tor".to_string()),
        std::io::sink(),
        None,
        Sealed::Opened(Some(Keys {
            cryptors: &cryptors,
            generation: Some(ks.generation),
        })),
    )
    .unwrap();
    assert_eq!(extraction.sealed_records, 2);
    assert_eq!(
        extraction.decode_failures, 2,
        "{:?}",
        extraction.failed_keys
    );
    assert!(
        extraction
            .failed_keys
            .iter()
            .any(|k| k.contains(&relocated.to_string()) && k.contains("does not open")),
        "{:?}",
        extraction.failed_keys
    );
}

#[test]
fn a_source_sentinel_ahead_of_its_keystore_refuses_the_extraction() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "uma", &["plain"]);
    let (ks, _) = seal_rocks_user(base, "uma");
    {
        let db = rocksdb::DB::open_cf(
            &rocksdb::Options::default(),
            base.join("uma/storage"),
            ["default", "kv", "memory_index"],
        )
        .unwrap();
        let index = db.cf_handle("memory_index").unwrap();
        db.put_cf(index, GENERATION_SENTINEL, 5u64.to_le_bytes())
            .unwrap();
        db.flush().unwrap();
    }
    let cryptors = record_keys(&ks);
    let err = extract_with(
        base,
        StorageBackend::RocksDb,
        &Unit::User("uma".to_string()),
        std::io::sink(),
        None,
        Sealed::Opened(Some(Keys {
            cryptors: &cryptors,
            generation: Some(ks.generation),
        })),
    )
    .expect_err("a keystore older than the store's sentinel");
    assert!(format!("{err:#}").contains("rollback"), "{err:#}");
}

#[cfg(all(feature = "rocksdb", feature = "storage-redb"))]
mod round_trip {
    use super::*;
    use crate::graph_memory::{EdgeSource, EdgeTier, LtpStatus, RelationType, RelationshipEdge};
    use crate::storage::boot_guard::{refuse_stranded_data, BackendCensus};
    use crate::storage::redb::{open_graph_database, RedbColumnStore};

    const TENANT: &str = "wanda";
    const CORPUS_SIZE: usize = 320;
    const ENTITY_COUNT: usize = 40;
    const KV_ROWS: usize = 30;
    const AUDIT_ENTRIES: usize = 150;
    const TODO_ROWS: usize = 8;
    const FACT_WATERMARK: i64 = 1_726_000_000_000;

    type Row = (String, Vec<u8>, Vec<u8>);
    type Canonical = BTreeMap<Unit, (u64, String, Vec<Row>)>;

    fn opposite(backend: StorageBackend) -> StorageBackend {
        match backend {
            StorageBackend::RocksDb => StorageBackend::Redb,
            StorageBackend::Redb => StorageBackend::RocksDb,
        }
    }

    fn corpus_memories() -> Vec<Memory> {
        (0..CORPUS_SIZE)
            .map(|i| {
                memory(
                    &format!("corpus memory {i} on subject {}", i % 17),
                    &format!("subject-{}", i % 17),
                )
            })
            .collect()
    }

    fn kv_rows() -> Vec<(Vec<u8>, Vec<u8>)> {
        (0..KV_ROWS)
            .map(|i| {
                (
                    format!("dial-{i:02}").into_bytes(),
                    format!("value-{i}").into_bytes(),
                )
            })
            .collect()
    }

    fn audit_rows() -> Vec<(Vec<u8>, Vec<u8>)> {
        (0..AUDIT_ENTRIES)
            .map(|i| {
                (
                    format!("{TENANT}:{i:020}").into_bytes(),
                    format!("event-{i}").into_bytes(),
                )
            })
            .collect()
    }

    fn todo_rows() -> Vec<(Vec<u8>, Vec<u8>)> {
        (0..TODO_ROWS)
            .map(|i| {
                (
                    format!("todo-{i}").into_bytes(),
                    format!("finish migration step {i}").into_bytes(),
                )
            })
            .collect()
    }

    fn edge_between(from: uuid::Uuid, to: uuid::Uuid, strength: f32) -> RelationshipEdge {
        let now = chrono::Utc::now();
        RelationshipEdge {
            uuid: uuid::Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            relation_type: RelationType::RelatedTo,
            strength,
            created_at: now,
            valid_at: now,
            invalidated_at: None,
            source_episode_id: None,
            context: "round trip corpus".to_string(),
            last_activated: now,
            activation_count: 3,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L1Working,
            created_by: EdgeSource::CoOccurrence,
            entity_confidence: None,
            forward_strength: strength,
            backward_strength: strength,
            strengthening_sources: Default::default(),
            last_corroborated: None,
        }
    }

    fn write_graph(graph: &GraphMemory) {
        let ids: Vec<uuid::Uuid> = (0..ENTITY_COUNT)
            .map(|i| graph.add_entity(entity(&format!("node{i}x"))).unwrap())
            .collect();
        for pair in ids.windows(2) {
            graph
                .add_relationship(edge_between(pair[0], pair[1], 0.6))
                .unwrap();
        }
    }

    fn fill_primary(store: &impl PrimaryMemoryStore, memories: &[Memory]) {
        for (i, m) in memories.iter().enumerate() {
            store.put_memory(m).unwrap();
            store
                .put_vector_mapping(&m.id, &mapping(i as u32 * 2))
                .unwrap();
        }
        for (key, value) in kv_rows() {
            store.key_value_store().put_bytes(&key, &value).unwrap();
        }
        store.set_fact_watermark(TENANT, FACT_WATERMARK);
    }

    fn rocks_origin(base: &Path) -> Vec<Memory> {
        let dir = base.join(TENANT);
        let memories = corpus_memories();
        let store = RocksDbPrimaryMemoryStore::open(&dir, None).unwrap();
        fill_primary(&store, &memories);
        store.flush().unwrap();
        drop(store);

        let graph = GraphMemory::new(&dir.join("graph"), None).unwrap();
        write_graph(&graph);
        drop(graph);

        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let shared =
            rocksdb::DB::open_cf(&opts, base.join("shared"), ["default", "audit", "todos"])
                .unwrap();
        for (cf, rows) in [("audit", audit_rows()), ("todos", todo_rows())] {
            let handle = shared.cf_handle(cf).unwrap();
            for (key, value) in rows {
                shared.put_cf(handle, key, value).unwrap();
            }
        }
        shared.flush().unwrap();
        memories
    }

    fn write_redb_table(path: &Path, table: &str, rows: &[(Vec<u8>, Vec<u8>)]) {
        let db = redb::Database::create(path).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut handle = txn.open_table(RawTable::new(table)).unwrap();
            for (key, value) in rows {
                handle.insert(key.as_slice(), value.as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    fn redb_origin(base: &Path) -> Vec<Memory> {
        let dir = base.join(TENANT);
        let memories = corpus_memories();
        let store = RedbPrimaryMemoryStore::open(&dir).unwrap();
        fill_primary(&store, &memories);
        drop(store);

        let graph_db = open_graph_database(&dir.join("graph")).unwrap();
        let graph =
            GraphMemory::with_store(std::sync::Arc::new(RedbColumnStore::new(graph_db))).unwrap();
        write_graph(&graph);
        drop(graph);

        write_redb_table(&base.join("audit.redb"), "audit", &audit_rows());
        write_redb_table(&base.join("shared.redb"), "todos", &todo_rows());
        memories
    }

    fn build_origin(base: &Path, backend: StorageBackend) -> Vec<Memory> {
        match backend {
            StorageBackend::RocksDb => rocks_origin(base),
            StorageBackend::Redb => redb_origin(base),
        }
    }

    /// Every unit's frame count, digest and frames in dump order, read through the same
    /// extraction migrate verifies with.
    fn canonical(base: &Path, backend: StorageBackend) -> Canonical {
        resolve_units(base, backend, &Scope::All)
            .unwrap()
            .into_iter()
            .map(|unit| {
                let mut dump = Vec::new();
                let extraction = extract(base, backend, &unit, &mut dump).unwrap();
                assert_eq!(
                    extraction.decode_failures, 0,
                    "{unit}: {:?}",
                    extraction.failed_keys
                );
                let rows = portable::read_all(dump.as_slice())
                    .unwrap()
                    .into_iter()
                    .map(|r| (r.cf, r.key, r.value))
                    .collect();
                (unit, (extraction.frames, extraction.digest, rows))
            })
            .collect()
    }

    fn rows_in(rows: &[Row], keyspace: &str) -> usize {
        rows.iter().filter(|(cf, _, _)| cf == keyspace).count()
    }

    fn assert_corpus_is_fully_represented(state: &Canonical) {
        let (_, _, user) = &state[&Unit::User(TENANT.to_string())];
        assert_eq!(rows_in(user, "primary/memories"), CORPUS_SIZE);
        assert_eq!(rows_in(user, "primary/vector_mappings"), CORPUS_SIZE);
        assert!(rows_in(user, "primary/kv") >= KV_ROWS);
        let graph_rows = user
            .iter()
            .filter(|(cf, _, _)| cf.starts_with("graph/"))
            .count();
        assert!(
            graph_rows >= ENTITY_COUNT + ENTITY_COUNT - 1,
            "the graph contributed {graph_rows} rows"
        );
        let (_, _, shared) = &state[&Unit::Shared];
        assert_eq!(rows_in(shared, "audit/audit"), AUDIT_ENTRIES);
        assert_eq!(rows_in(shared, "shared/todos"), TODO_ROWS);
    }

    fn engine_paths(base: &Path, backend: StorageBackend) -> Vec<PathBuf> {
        let user = base.join(TENANT);
        match backend {
            StorageBackend::RocksDb => vec![
                user.join("storage"),
                user.join("graph/graph"),
                base.join("shared"),
            ],
            StorageBackend::Redb => vec![
                user.join("storage.redb"),
                user.join("graph/graph.redb"),
                user.join(KEYSTORE),
                base.join("shared.redb"),
                base.join("audit.redb"),
            ],
        }
    }

    fn engine_files(base: &Path, backend: StorageBackend) -> BTreeMap<PathBuf, String> {
        let mut files = BTreeMap::new();
        for path in engine_paths(base, backend) {
            if path.is_dir() {
                for (relative, digest) in snapshot(&path) {
                    files.insert(path.join(relative), digest);
                }
            } else if path.is_file() {
                let digest: [u8; 32] = Sha256::digest(std::fs::read(&path).unwrap()).into();
                files.insert(path, portable::hex(&digest));
            }
        }
        files
    }

    fn copy_tree(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_tree(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    /// A copy of `from` without the `source` engine, so the next hop starts from only
    /// what the previous migration produced.
    fn carry_over_target_only(from: &Path, to: &Path, source: StorageBackend) {
        copy_tree(from, to);
        for path in engine_paths(to, source) {
            if path.is_dir() {
                std::fs::remove_dir_all(&path).unwrap();
            } else if path.is_file() {
                std::fs::remove_file(&path).unwrap();
            }
        }
    }

    fn users() -> Vec<String> {
        vec![TENANT.to_string()]
    }

    fn assert_boot_refused(
        outcome: anyhow::Result<BackendCensus>,
        selected: StorageBackend,
        holder: StorageBackend,
    ) {
        let message = format!("{:#}", outcome.expect_err("an empty engine must not boot"));
        for expected in [
            "refusing to serve".to_string(),
            format!("the {selected} store is empty"),
            TENANT.to_string(),
            format!("VELD_STORAGE_BACKEND={holder}"),
            format!("veld migrate --from {holder} --to {selected}"),
        ] {
            assert!(
                message.contains(&expected),
                "{expected:?} not in: {message}"
            );
        }
    }

    fn assert_both_engines_boot(base: &Path) {
        for backend in [StorageBackend::Redb, StorageBackend::RocksDb] {
            let census = refuse_stranded_data(base, backend, &users())
                .unwrap_or_else(|e| panic!("{backend} refused to boot: {e:#}"));
            assert_eq!((census.rocksdb, census.redb), (1, Some(1)));
        }
    }

    fn assert_serves_corpus(base: &Path, backend: StorageBackend, expected: &[Memory]) {
        fn check(store: &impl PrimaryMemoryStore, expected: &[Memory]) {
            let served = store.get_all().unwrap();
            assert_eq!(sorted_ids(&served), sorted_ids(expected));
            let content: std::collections::HashMap<_, _> = served
                .iter()
                .map(|m| (m.id.0, m.experience.content.clone()))
                .collect();
            for m in expected {
                assert_eq!(content[&m.id.0], m.experience.content);
                assert!(
                    store.get_vector_mapping(&m.id).unwrap().is_some(),
                    "mapping of {} lost",
                    m.id.0
                );
            }
            assert_eq!(
                store.key_value_store().get_bytes(b"dial-07").unwrap(),
                Some(b"value-7".to_vec())
            );
            assert_eq!(store.get_fact_watermark(TENANT), Some(FACT_WATERMARK));
        }
        let dir = base.join(TENANT);
        match backend {
            StorageBackend::RocksDb => check(
                &RocksDbPrimaryMemoryStore::open(&dir, None).unwrap(),
                expected,
            ),
            StorageBackend::Redb => check(&RedbPrimaryMemoryStore::open(&dir).unwrap(), expected),
        }
    }

    fn migrate_and_check_hop(
        base: &Path,
        from: StorageBackend,
        to: StorageBackend,
        origin: &Canonical,
    ) {
        let source_before = engine_files(base, from);
        let report = migrate(base, from, to, &Scope::All, false, None)
            .unwrap_or_else(|e| panic!("{from} -> {to}: {e:#}"));
        assert_eq!(report.imported.len(), origin.len(), "{from} -> {to}");
        for imported in &report.imported {
            assert_eq!(
                (imported.verified_frames, &imported.verified_digest),
                (imported.frames, &imported.digest),
                "{from} -> {to}: {} did not verify",
                imported.unit
            );
            let (frames, digest, _) = &origin[&imported.unit];
            assert_eq!(
                (imported.frames, &imported.digest),
                (*frames, digest),
                "{from} -> {to}: {} drifted from the origin",
                imported.unit
            );
        }
        assert_eq!(
            engine_files(base, from),
            source_before,
            "{from} -> {to} modified its source"
        );
    }

    fn round_trip(origin_backend: StorageBackend) {
        let other = opposite(origin_backend);
        let tmp = tempfile::tempdir().unwrap();
        let first = tmp.path().join("first");
        let second = tmp.path().join("second");
        let memories = build_origin(&first, origin_backend);

        assert_boot_refused(
            refuse_stranded_data(&first, other, &users()),
            other,
            origin_backend,
        );
        refuse_stranded_data(&first, origin_backend, &users()).expect("the holder boots");

        let origin = canonical(&first, origin_backend);
        assert_corpus_is_fully_represented(&origin);

        migrate_and_check_hop(&first, origin_backend, other, &origin);
        assert_eq!(
            canonical(&first, other),
            origin,
            "{origin_backend} -> {other} changed the logical contents"
        );
        carry_over_target_only(&first, &second, origin_backend);
        assert_both_engines_boot(&first);
        assert_serves_corpus(&first, origin_backend, &memories);
        assert_serves_corpus(&first, other, &memories);

        assert_boot_refused(
            refuse_stranded_data(&second, origin_backend, &users()),
            origin_backend,
            other,
        );
        refuse_stranded_data(&second, other, &users()).expect("the migrated engine boots");

        migrate_and_check_hop(&second, other, origin_backend, &origin);
        assert_eq!(
            canonical(&second, origin_backend),
            origin,
            "{origin_backend} -> {other} -> {origin_backend} did not return the origin"
        );
        assert_both_engines_boot(&second);
        assert_serves_corpus(&second, origin_backend, &memories);
        assert_serves_corpus(&second, other, &memories);
    }

    #[test]
    fn rocksdb_to_redb_to_rocksdb_returns_the_origin_with_every_hop_verified() {
        round_trip(StorageBackend::RocksDb);
    }

    #[test]
    fn redb_to_rocksdb_to_redb_returns_the_origin_with_every_hop_verified() {
        round_trip(StorageBackend::Redb);
    }

    #[test]
    fn a_dry_run_changes_no_byte_of_the_data_directory_in_either_direction() {
        for from in [StorageBackend::RocksDb, StorageBackend::Redb] {
            let to = opposite(from);
            let tmp = tempfile::tempdir().unwrap();
            build_origin(tmp.path(), from);
            let before = snapshot(tmp.path());

            let plan = migrate(tmp.path(), from, to, &Scope::All, true, None).unwrap();
            assert!(!plan.blocked(), "{from} -> {to}: {:?}", plan.units);
            assert!(plan.imported.is_empty());
            assert_eq!(plan.units.len(), 2);
            for unit in &plan.units {
                assert!(unit.source.as_ref().unwrap().records > 0);
                assert_eq!(unit.target.as_ref().unwrap().records, 0, "{from} -> {to}");
            }

            assert_eq!(
                snapshot(tmp.path()),
                before,
                "{from} -> {to}: a dry run changed the data directory"
            );
        }
    }
}

#[cfg(all(feature = "rocksdb", feature = "storage-redb"))]
mod sealed_redb_dump {
    use super::*;

    fn seal_redb_user(base: &Path, id: &str) -> (Keystore, Vec<Row>) {
        use crate::keystore::KdfParams;
        let dir = base.join(id);
        std::fs::create_dir_all(&dir).unwrap();
        let mut ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        let before = RecordCryptors::from_keystore(&ks, &kek).unwrap();
        ks.rotate_dek(&kek).unwrap();
        let after = RecordCryptors::from_keystore(&ks, &kek).unwrap();
        std::fs::write(dir.join(KEYSTORE), ks.to_json().unwrap()).unwrap();

        let mut rows = Vec::new();
        for (cryptors, n) in [(&before, 0), (&after, 1)] {
            let m = memory(&format!("{SEALED_SECRET} {n}"), "sealed");
            let plain = bincode::serde::encode_to_vec(&m, bincode::config::standard()).unwrap();
            let sealed = cryptors
                .active()
                .encrypt_record(&plain, m.id.0.as_bytes())
                .unwrap();
            rows.push((m.id.0.as_bytes().to_vec(), sealed));
        }
        rows.sort();
        let database = redb::Database::create(dir.join("storage.redb")).unwrap();
        let txn = database.begin_write().unwrap();
        {
            let mut memories = txn.open_table(RawTable::new("memories")).unwrap();
            for (key, value) in &rows {
                memories.insert(key.as_slice(), value.as_slice()).unwrap();
            }
            let mut default = txn.open_table(RawTable::new("default")).unwrap();
            default
                .insert(GENERATION_SENTINEL, 1u64.to_le_bytes().as_slice())
                .unwrap();
        }
        txn.commit().unwrap();
        (ks, rows)
    }

    #[test]
    fn a_migration_dump_of_a_redb_store_carries_sealed_records_as_stored_and_no_plaintext() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let (ks, sealed) = seal_redb_user(base, "vera");
        let cryptors = record_keys(&ks);

        let dump = tmp.path().join("vera.veldport");
        let (stats, _) = export_unit(
            base,
            StorageBackend::Redb,
            &Unit::User("vera".to_string()),
            &dump,
            None,
            Sealed::Opened(Some(Keys {
                cryptors: &cryptors,
                generation: Some(ks.generation),
            })),
        )
        .expect("a migration export of an encrypted redb store");
        assert_eq!(stats.sealed_records, 2);
        assert_eq!(stats.decode_failures, 0);

        let bytes = std::fs::read(&dump).unwrap();
        assert!(
            !contains(&bytes, SEALED_SECRET),
            "a sealed record reached the dump opened"
        );
        let records = portable::read_all(bytes.as_slice()).unwrap();
        for (key, value) in &sealed {
            let carried = records
                .iter()
                .find(|r| r.cf == "primary/memories" && &r.key == key)
                .expect("sealed record carried");
            assert_eq!(
                &carried.value, value,
                "a sealed record was not carried as stored"
            );
        }
        let sentinel: Vec<_> = records
            .iter()
            .filter(|r| r.key == GENERATION_SENTINEL)
            .collect();
        assert_eq!(sentinel.len(), 1);
        assert_eq!(sentinel[0].cf, "primary/default");
        assert_eq!(sentinel[0].value, 1u64.to_le_bytes());
    }
}
