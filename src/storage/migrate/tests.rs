use super::*;
use crate::graph_memory::{EntityLabel, EntityNode, GraphMemory, PiiClassification};
use crate::memory::storage::{SearchCriteria, VectorMappingEntry};
use crate::memory::types::{Experience, ExperienceType, Memory, MemoryId, MemoryTier};
use crate::storage::legacy_rocksdb::RocksDbPrimaryMemoryStore;
use crate::storage::redb::RedbPrimaryMemoryStore;
use crate::storage::PrimaryMemoryStore;
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
        0.5, 0, now, now, false, MemoryTier::LongTerm, Vec::new(), 1.0,
        None, None, None, None, 0.0, None, None, 1, Vec::new(), Vec::new(),
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
    let memories: Vec<Memory> = tags.iter().map(|t| memory(&format!("about {t}"), t)).collect();
    for (i, m) in memories.iter().enumerate() {
        store.put_memory(m).unwrap();
        store.put_vector_mapping(&m.id, &mapping(i as u32 * 10)).unwrap();
    }
    store.set_fact_watermark(id, 1_726_000_000_000);
    store.key_value_store().put_bytes(b"dials", b"tuned").unwrap();
    store.flush().unwrap();
    drop(store);
    let graph = GraphMemory::new(&dir.join("graph"), None).unwrap();
    let uuid = graph.add_entity(entity("Rust")).unwrap();
    drop(graph);
    (memories, uuid)
}

fn rocks_rows(dir: &Path) -> u64 {
    let store = readonly::open_rocks(dir).unwrap().expect("store present");
    let mut n = 0;
    for cf in &store.column_families {
        let handle = store.db.cf_handle(cf).unwrap();
        n += store.db.iterator_cf(handle, rocksdb::IteratorMode::Start).count() as u64;
    }
    n
}

/// Every file under `root` with a content hash, so "unchanged" is measured, not assumed.
fn snapshot(root: &Path) -> BTreeMap<PathBuf, String> {
    fn walk(dir: &Path, root: &Path, out: &mut BTreeMap<PathBuf, String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, root, out);
            } else {
                let bytes = std::fs::read(&path).unwrap();
                let digest: [u8; 32] = Sha256::digest(&bytes).into();
                out.insert(path.strip_prefix(root).unwrap().to_path_buf(), portable::hex(&digest));
            }
        }
    }
    let mut out = BTreeMap::new();
    if root.exists() {
        walk(root, root, &mut out);
    }
    out
}

fn plant_undecodable_memory(base: &Path, id: &str) -> uuid::Uuid {
    plant_undecodable_rocks_memory(&base.join(id))
}

fn logical_dump(path: &Path, frames: &[(&str, &[u8], &[u8])]) {
    let mut buf = Vec::new();
    let mut w = PortableWriter::new(&mut buf).unwrap();
    for (cf, key, value) in frames {
        w.write_record(&PortableRecord { cf: cf.to_string(), key: key.to_vec(), value: value.to_vec() })
            .unwrap();
    }
    w.finish().unwrap();
    std::fs::write(path, buf).unwrap();
}

fn user(id: &str) -> Scope {
    Scope::User(id.to_string())
}

#[test]
fn rocks_to_redb_is_read_back_by_the_redb_stores_themselves() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    let (memories, entity_id) = rocks_user(base, "alice", &["rust", "redb", "cutover"]);

    let report = migrate(base, StorageBackend::RocksDb, StorageBackend::Redb, &user("alice"), false, None)
        .expect("migrate");
    assert_eq!(report.imported.len(), 1);
    let imported = &report.imported[0];
    assert_eq!((imported.verified_frames, &imported.verified_digest), (imported.frames, &imported.digest));

    let redb = RedbPrimaryMemoryStore::open(&base.join("alice")).unwrap();
    assert_eq!(sorted_ids(&redb.get_all().unwrap()), sorted_ids(&memories));
    for m in &memories {
        assert_eq!(redb.get(&m.id).unwrap().experience.content, m.experience.content);
    }
    let got = redb.get_vector_mapping(&memories[1].id).unwrap().expect("mapping carried");
    assert_eq!(got.modalities.len(), 1);
    assert_eq!(redb.get_fact_watermark("alice"), Some(1_726_000_000_000));
    assert_eq!(redb.key_value_store().get_bytes(b"dials").unwrap(), Some(b"tuned".to_vec()));
    drop(redb);

    let graph_db = crate::storage::redb::open_graph_database(&base.join("alice/graph")).unwrap();
    let graph = GraphMemory::with_store(std::sync::Arc::new(
        crate::storage::redb::RedbColumnStore::new(graph_db),
    ))
    .unwrap();
    assert_eq!(graph.get_entity(&entity_id).unwrap().map(|e| e.name), Some("Rust".to_string()));
}

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

    migrate(base, StorageBackend::Redb, StorageBackend::RocksDb, &user("bob"), false, None).expect("migrate");

    let rocks = RocksDbPrimaryMemoryStore::open(&dir, None).unwrap();
    assert_eq!(rocks.get_all().unwrap().len(), 2);
    let found = rocks.search(SearchCriteria::ByTags(vec!["findable".into()])).unwrap();
    assert_eq!(sorted_ids(&found), vec![tagged.id.0], "memory_index was not rebuilt on the rocksdb target");
}

#[test]
fn shared_stores_cross_backends_with_audit_in_its_own_file() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    {
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let db = rocksdb::DB::open_cf(&opts, base.join("shared"), ["default", "audit", "todos"]).unwrap();
        db.put_cf(db.cf_handle("audit").unwrap(), b"alice:00000000000000000001", b"event").unwrap();
        db.put_cf(db.cf_handle("todos").unwrap(), b"todo-1", b"write the migration").unwrap();
        db.flush().unwrap();
    }

    let report = migrate(base, StorageBackend::RocksDb, StorageBackend::Redb, &Scope::All, false, None).unwrap();
    assert_eq!(report.imported.iter().map(|r| r.unit.clone()).collect::<Vec<_>>(), vec![Unit::Shared]);

    let read_row = |file: &str, table: &str, key: &[u8]| -> Option<Vec<u8>> {
        let db = redb::Database::open(base.join(file)).unwrap();
        let read = db.begin_read().unwrap();
        let t = read.open_table(RawTable::new(table)).unwrap();
        let v = redb::ReadableTable::get(&t, key).unwrap().map(|v| v.value().to_vec());
        v
    };
    assert_eq!(read_row("audit.redb", "audit", b"alice:00000000000000000001"), Some(b"event".to_vec()));
    assert_eq!(read_row("shared.redb", "todos", b"todo-1"), Some(b"write the migration".to_vec()));
}

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
    assert!(msg.contains(&bad.to_string()), "the failing key must be named: {msg}");
    assert_eq!(snapshot(&out), BTreeMap::new(), "a failed export left files behind");

    let plan = migrate(base, StorageBackend::RocksDb, StorageBackend::Redb, &user("carol"), true, None).unwrap();
    assert!(plan.blocked());
    let source = plan.units[0].source.as_ref().unwrap();
    assert_eq!(source.decode_failures, 1);
    assert_eq!(source.records, 2 * 2 + 2 + rocks_rows(&base.join("carol/graph/graph")));
    let err = migrate(base, StorageBackend::RocksDb, StorageBackend::Redb, &user("carol"), false, None)
        .expect_err("a real migration must refuse");
    assert!(format!("{err:#}").contains("refused before writing anything"));
    assert!(!base.join("carol/storage.redb").exists(), "a refused migration created the target");
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
    let seen = extract(base, StorageBackend::RocksDb, &Unit::User("cory".into()), std::io::sink()).unwrap();
    assert_eq!(seen.decode_failures, 0, "{:?}", seen.failed_keys);
}

#[test]
fn an_import_whose_rows_do_not_land_as_declared_fails_verification() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    let dump = tmp.path().join("dup.veldport");
    // Two frames for one key: the codec accepts it, the store keeps one row.
    logical_dump(&dump, &[
        (META, b"format", FORMAT.as_bytes()),
        (META, b"unit", b"user"),
        (META, b"user_id", b"dave"),
        ("primary/kv", b"k", b"first"),
        ("primary/kv", b"k", b"second"),
    ]);
    let err = import(base, StorageBackend::Redb, &dump).expect_err("a mismatch must not report success");
    let msg = format!("{err:#}");
    assert!(msg.contains("verification FAILED"), "{msg}");
    assert!(msg.contains("declares 5 frames"), "{msg}");
    assert!(msg.contains("re-derives 4 frames"), "{msg}");
}

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

#[test]
fn dry_run_writes_nothing_on_either_side() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    let (memories, _) = rocks_user(base, "frank", &["p", "q", "r"]);
    // An empty redb file already present, as on the live box.
    drop(RedbPrimaryMemoryStore::open(&base.join("frank")).unwrap());
    let before = snapshot(base);

    let plan = migrate(base, StorageBackend::RocksDb, StorageBackend::Redb, &user("frank"), true, None).unwrap();
    assert!(!plan.blocked(), "{:?}", plan.units[0].blocked);
    let source = plan.units[0].source.as_ref().unwrap();
    let graph_rows = rocks_rows(&base.join("frank/graph/graph"));
    assert_eq!(source.records, memories.len() as u64 * 2 + 2 + graph_rows);
    assert!(source.derived_rows_not_carried > 0);
    assert_eq!(plan.units[0].target.as_ref().unwrap().records, 0);
    assert!(plan.imported.is_empty());

    assert_eq!(snapshot(base), before, "a dry run changed the data directory");
}

#[test]
fn a_dry_run_never_creates_an_absent_target() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "gina", &["only"]);
    let before = snapshot(base);
    migrate(base, StorageBackend::RocksDb, StorageBackend::Redb, &Scope::All, true, None).unwrap();
    assert_eq!(snapshot(base), before);
    assert!(!base.join("gina/storage.redb").exists());
}

#[test]
fn migrate_leaves_the_source_byte_identical() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "hank", &["one", "two"]);
    let source_before = snapshot(&base.join("hank/storage"));
    let graph_before = snapshot(&base.join("hank/graph/graph"));

    migrate(base, StorageBackend::RocksDb, StorageBackend::Redb, &user("hank"), false, None).unwrap();

    assert_eq!(snapshot(&base.join("hank/storage")), source_before, "the rocksdb source was modified");
    assert_eq!(snapshot(&base.join("hank/graph/graph")), graph_before, "the rocksdb graph source was modified");
}

#[test]
fn a_redb_source_is_read_without_touching_its_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("ivy");
    RedbPrimaryMemoryStore::open(&dir).unwrap().put_memory(&memory("m", "t")).unwrap();
    let before = snapshot(tmp.path());
    let seen = extract(tmp.path(), StorageBackend::Redb, &Unit::User("ivy".into()), std::io::sink()).unwrap();
    assert_eq!(seen.records, 1);
    assert_eq!(snapshot(tmp.path()), before, "reading a redb store wrote to it");
}

#[test]
fn a_redb_store_held_by_a_writer_is_refused_not_read_mid_write() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("jack");
    let live = RedbPrimaryMemoryStore::open(&dir).unwrap();
    live.put_memory(&memory("m", "t")).unwrap();
    let err = extract(tmp.path(), StorageBackend::Redb, &Unit::User("jack".into()), std::io::sink())
        .expect_err("a store held open for writing must refuse");
    assert!(format!("{err:#}").contains("held open for writing"), "{err:#}");
    drop(live);
}

#[test]
fn a_populated_target_refuses_the_whole_migration_before_writing() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();
    rocks_user(base, "kate", &["a"]);
    rocks_user(base, "liam", &["b"]);
    RedbPrimaryMemoryStore::open(&base.join("liam")).unwrap().put_memory(&memory("already", "here")).unwrap();
    let before = snapshot(base);

    let err = migrate(base, StorageBackend::RocksDb, StorageBackend::Redb, &Scope::All, false, None)
        .expect_err("a populated target must refuse");
    assert!(format!("{err:#}").contains("liam"), "{err:#}");
    assert_eq!(snapshot(base), before, "kate was migrated even though liam refused");
}

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
    let err = export(base, StorageBackend::RocksDb, &user("pia"), &tmp.path().join("out"), None)
        .expect_err("encrypted stores must refuse");
    assert!(format!("{err:#}").contains("record-level encryption"), "{err:#}");
}

#[test]
fn a_named_user_with_no_store_is_an_error_not_an_empty_dump() {
    let tmp = tempfile::tempdir().unwrap();
    let err = export(tmp.path(), StorageBackend::RocksDb, &user("nobody"), &tmp.path().join("out"), None)
        .expect_err("absent user");
    assert!(format!("{err:#}").contains("no rocksdb store for user 'nobody'"));
}

#[test]
fn quarantine_sets_undecodable_records_aside_byte_for_byte_and_accounts_for_them() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("data");
    let (memories, _) = rocks_user(&base, "quinn", &["kept", "also kept"]);
    let bad = plant_undecodable_memory(&base, "quinn");
    let source = extract(&base, StorageBackend::RocksDb, &Unit::User("quinn".into()), std::io::sink()).unwrap();
    assert_eq!(source.decode_failures, 1);

    let out = tmp.path().join("export");
    let qdir = tmp.path().join("quarantine");
    let reports = export(&base, StorageBackend::RocksDb, &user("quinn"), &out, Some(&qdir)).expect("quarantined export");
    let r = &reports[0];
    assert_eq!(r.quarantined, 1);
    assert_eq!(r.records + r.quarantined, source.source_records());
    assert_eq!(r.records, source.records);
    let q = r.quarantine.as_ref().expect("quarantine reported");
    assert_eq!(q.ids, vec![format!("primary/memories {bad}")]);

    let (trailer_frames, _) = portable::read_trailer(&r.path).unwrap();
    assert_eq!(trailer_frames, r.frames);
    let exported = portable::read_all(std::fs::File::open(&r.path).unwrap()).unwrap();
    assert!(exported.iter().all(|rec| rec.key != bad.as_bytes()), "a quarantined record was exported");

    let held = portable::read_all(std::fs::File::open(&q.dump).unwrap()).unwrap();
    assert_eq!(held.len(), 1);
    assert_eq!((held[0].cf.as_str(), held[0].key.as_slice(), held[0].value.as_slice()),
        ("primary/memories", &bad.as_bytes()[..], UNDECODABLE_VALUE));
    let listing: serde_json::Value = serde_json::from_slice(&std::fs::read(&q.listing).unwrap()).unwrap();
    let digest: [u8; 32] = Sha256::digest(UNDECODABLE_VALUE).into();
    assert_eq!(listing["entries"][0]["value_sha256"], portable::hex(&digest));

    let target = tmp.path().join("target");
    std::fs::create_dir_all(&target).unwrap();
    let imported = import(&target, StorageBackend::Redb, &out).expect("the shortened dump still verifies");
    assert_eq!(imported[0].verified_frames, r.frames);
    let redb = RedbPrimaryMemoryStore::open(&target.join("quinn")).unwrap();
    assert_eq!(sorted_ids(&redb.get_all().unwrap()), sorted_ids(&memories));
}

#[test]
fn migrate_with_quarantine_migrates_the_rest_and_reports_what_it_set_aside() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("data");
    let (memories, _) = rocks_user(&base, "rhea", &["a", "b", "c"]);
    let bad = plant_undecodable_memory(&base, "rhea");
    let source_before = snapshot(&base.join("rhea/storage"));
    let qdir = tmp.path().join("quarantine");

    let plan = migrate(&base, StorageBackend::RocksDb, StorageBackend::Redb, &user("rhea"), true, Some(&qdir)).unwrap();
    assert!(!plan.blocked(), "{:?}", plan.units[0].blocked);
    assert!(!qdir.exists(), "a dry run wrote a quarantine directory");

    let report = migrate(&base, StorageBackend::RocksDb, StorageBackend::Redb, &user("rhea"), false, Some(&qdir))
        .expect("migrate with quarantine");
    assert_eq!(report.quarantined.len(), 1);
    assert_eq!(report.quarantined[0].ids, vec![format!("primary/memories {bad}")]);
    assert_eq!(report.imported[0].records + report.quarantined[0].records,
        plan.units[0].source.as_ref().unwrap().source_records());
    let redb = RedbPrimaryMemoryStore::open(&base.join("rhea")).unwrap();
    assert_eq!(sorted_ids(&redb.get_all().unwrap()), sorted_ids(&memories));
    assert_eq!(snapshot(&base.join("rhea/storage")), source_before, "the source was modified");
}

#[test]
fn the_planted_record_is_one_the_raw_utf8_heuristic_would_have_fabricated_a_memory_from() {
    // Guards the plant: the runtime reader accepts these bytes via its last-resort arm,
    // so a test that planted bytes nothing at all accepts would not cover that arm.
    assert!(crate::memory::storage::portable_memory_value(UNDECODABLE_VALUE).is_err());
    assert!(crate::memory::storage::deserialize_memory_for_test(UNDECODABLE_VALUE).is_ok());
}
