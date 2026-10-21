//! Storage-engine comparison for ADR 0007: the same corpus and the same operations on
//! every engine this build carries.
//!
//! Each operation is one Criterion group. Within a group the benchmark id is
//! `{engine}-{mode}/{corpus size}` for operations on memory records, where `mode` is
//! `plaintext` or `encrypted` (encryption-v2), and `{engine}/{corpus size}` for the
//! graph, key-value and batch operations, which encryption-v2 does not seal.
//!
//! Every fixture is built before the first measurement, and every read group runs
//! before any write group, so no read is measured over a store a write benchmark has
//! touched. Write benchmarks undo their own insert outside the timed section, so the
//! corpus stays at its nominal size.
//!
//! Every store opens at the process's `StorageTuning`, as the server's do, so both
//! engines run at one durability class: `eventual` unless `VELD_STORAGE_DURABILITY`
//! says `immediate`. The run logs the class and cache sizes it measured at.
//!
//! Environment:
//! - `VELD_BENCH_CORPUS_SIZES` — comma-separated corpus sizes (default `1000,10000,100000`)
//! - `VELD_BENCH_ENGINES` — comma-separated engines (default: every engine built in)
//! - `VELD_STORAGE_DURABILITY` — `eventual` (default) or `immediate`, for both engines
//! - `VELD_REDB_CACHE_BYTES`, `VELD_ROCKSDB_WRITE_BUFFER_BYTES`,
//!   `VELD_ROCKSDB_BLOCK_CACHE_BYTES` — cache and memtable sizes, as on the server

mod storage_corpus;

use std::collections::HashSet;
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, Bencher, BenchmarkGroup, BenchmarkId, Criterion, SamplingMode,
    Throughput,
};
use tempfile::TempDir;
use uuid::Uuid;

use veld::graph_memory::{
    EdgeSource, EntityLabel, EntityNode, GraphMemory, PiiClassification, RelationType,
    RelationshipEdge,
};
use veld::memory::storage::SearchCriteria;
use veld::memory::{Memory, MemoryId};
use veld::storage::{BatchWriter, KeyValueStore, PrimaryMemoryStore, WriteBatch};

use storage_corpus::{load_corpus, CorpusGenerator, Engine, OpenStore, SplitMix64};

const CORPUS_SIZES_ENV: &str = "VELD_BENCH_CORPUS_SIZES";
const DEFAULT_CORPUS_SIZES: &[usize] = &[1_000, 10_000, 100_000];

const CORPUS_SEED: u64 = 0x5EED_C0A9_0001;
const WRITE_SEED: u64 = 0x5EED_C0A9_0002;
const READ_SEED: u64 = 0x5EED_C0A9_0003;
const KEY_VALUE_SEED: u64 = 0x5EED_C0A9_0004;
const GRAPH_SEED: u64 = 0x5EED_C0A9_0005;

const KV_VALUE_BYTES: usize = 256;
const KV_LOAD_BATCH: usize = 10_000;
const KV_VALUE_POOL: usize = 64;

const WRITE_BATCH_OPS: usize = 100;
/// The general-purpose family both engines carry; the key-value fixture fills it.
const WRITE_BATCH_FAMILY: &str = "kv";

/// One entity per four edges gives a mean degree of eight.
const EDGES_PER_ENTITY: usize = 4;
#[cfg(feature = "storage-redb")]
const GRAPH_LOAD_BATCH: usize = 20_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordMode {
    Plaintext,
    Encrypted,
}

impl RecordMode {
    fn name(self) -> &'static str {
        match self {
            RecordMode::Plaintext => "plaintext",
            RecordMode::Encrypted => "encrypted",
        }
    }
}

struct RecordStore {
    mode: RecordMode,
    store: Arc<dyn PrimaryMemoryStore>,
}

/// Everything one engine is measured on at one corpus size. The directories are
/// declared last so every handle into them closes before they are removed.
struct EngineFixture {
    engine: Engine,
    corpus_size: usize,
    ids: Vec<MemoryId>,
    records: Vec<RecordStore>,
    key_values: Arc<dyn KeyValueStore>,
    kv_keys: Vec<Vec<u8>>,
    batch_writer: Box<dyn BatchWriter>,
    graph: GraphMemory,
    entities: Vec<Uuid>,
    _directories: Vec<TempDir>,
}

impl EngineFixture {
    fn build(engine: Engine, memories: &[Memory]) -> Result<Self> {
        let corpus_size = memories.len();

        let plaintext_dir = TempDir::new()?;
        let plaintext = load_corpus(engine, plaintext_dir.path(), false, memories)?;
        let encrypted_dir = TempDir::new()?;
        let encrypted = load_corpus(engine, encrypted_dir.path(), true, memories)?;

        let key_values = plaintext.primary().key_value_store();
        let kv_keys = load_key_values(key_values.as_ref(), corpus_size)?;
        let batch_writer = batch_writer(&plaintext);

        let graph_dir = TempDir::new()?;
        let (graph, entities) = build_graph(engine, graph_dir.path(), corpus_size)?;

        Ok(Self {
            engine,
            corpus_size,
            ids: memories.iter().map(|memory| memory.id.clone()).collect(),
            records: vec![
                RecordStore {
                    mode: RecordMode::Plaintext,
                    store: plaintext.primary(),
                },
                RecordStore {
                    mode: RecordMode::Encrypted,
                    store: encrypted.primary(),
                },
            ],
            key_values,
            kv_keys,
            batch_writer,
            graph,
            entities,
            _directories: vec![plaintext_dir, encrypted_dir, graph_dir],
        })
    }
}

fn build_fixtures() -> Result<Vec<EngineFixture>> {
    storage_corpus::announce_tuning("storage_engine_benchmarks");
    let engines = Engine::selected();
    let sizes = storage_corpus::selected_sizes(CORPUS_SIZES_ENV, DEFAULT_CORPUS_SIZES);
    let mut fixtures = Vec::with_capacity(engines.len() * sizes.len());
    for &corpus_size in &sizes {
        let memories = CorpusGenerator::new(CORPUS_SEED).corpus(corpus_size);
        for &engine in &engines {
            eprintln!(
                "storage_engine_benchmarks: loading {corpus_size} memories into {}",
                engine.name()
            );
            let fixture = EngineFixture::build(engine, &memories).with_context(|| {
                format!("building the {} fixture of {corpus_size}", engine.name())
            })?;
            fixtures.push(fixture);
        }
    }
    Ok(fixtures)
}

/// A writer over the plaintext store's own database handle, as production builds one.
fn batch_writer(store: &OpenStore) -> Box<dyn BatchWriter> {
    match store {
        #[cfg(feature = "rocksdb")]
        OpenStore::RocksDb(store) => {
            Box::new(veld::storage::RocksDbBatchWriter::new(store.inner().db()))
        }
        #[cfg(feature = "storage-redb")]
        OpenStore::Redb(store) => Box::new(veld::storage::RedbBatchWriter::new(store.database())),
    }
}

fn random_bytes(rng: &mut SplitMix64, len: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(len + 8);
    while bytes.len() < len {
        bytes.extend_from_slice(&rng.next_u64().to_le_bytes());
    }
    bytes.truncate(len);
    bytes
}

fn load_key_values(store: &dyn KeyValueStore, count: usize) -> Result<Vec<Vec<u8>>> {
    let mut rng = SplitMix64::new(KEY_VALUE_SEED);
    let keys: Vec<Vec<u8>> = (0..count)
        .map(|ordinal| format!("bench:kv:{ordinal:08}").into_bytes())
        .collect();
    for chunk in keys.chunks(KV_LOAD_BATCH) {
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = chunk
            .iter()
            .map(|key| (key.clone(), random_bytes(&mut rng, KV_VALUE_BYTES)))
            .collect();
        store.put_many(&pairs)?;
    }
    for key in [keys.first(), keys.last()].into_iter().flatten() {
        ensure!(
            store.get_bytes(key)?.is_some(),
            "key-value fixture lost {}",
            String::from_utf8_lossy(key)
        );
    }
    Ok(keys)
}

fn corpus_relation(choice: usize) -> RelationType {
    match choice % 5 {
        0 => RelationType::RelatedTo,
        1 => RelationType::CoOccurs,
        2 => RelationType::Uses,
        3 => RelationType::PartOf,
        _ => RelationType::DependsOn,
    }
}

fn entity_node(ordinal: usize) -> EntityNode {
    let now = chrono::Utc::now();
    let (prefix, label) = match ordinal % 3 {
        0 => ("concept", EntityLabel::Concept),
        1 => ("technology", EntityLabel::Technology),
        _ => ("project", EntityLabel::Project),
    };
    EntityNode {
        uuid: Uuid::nil(),
        name: format!("{prefix}-{ordinal:06}"),
        labels: vec![label],
        created_at: now,
        last_seen_at: now,
        mention_count: 1,
        summary: String::new(),
        attributes: std::collections::HashMap::new(),
        name_embedding: None,
        salience: 0.5,
        is_proper_noun: false,
        pii_classification: PiiClassification::Clean,
    }
}

fn edge(from: Uuid, to: Uuid, relation_type: RelationType, strength: f32) -> RelationshipEdge {
    let now = chrono::Utc::now();
    RelationshipEdge {
        uuid: Uuid::nil(),
        from_entity: from,
        to_entity: to,
        relation_type,
        strength,
        created_at: now,
        valid_at: now,
        invalidated_at: None,
        source_episode_id: None,
        context: String::new(),
        last_activated: now,
        activation_count: 1,
        ltp_status: Default::default(),
        tier: Default::default(),
        activation_timestamps: None,
        entity_confidence: None,
        created_by: EdgeSource::CoOccurrence,
        forward_strength: strength,
        backward_strength: strength,
        strengthening_sources: Default::default(),
        last_corroborated: None,
    }
}

fn distinct_pair(rng: &mut SplitMix64, entities: &[Uuid]) -> (Uuid, Uuid) {
    let from = entities[rng.below(entities.len())];
    loop {
        let to = entities[rng.below(entities.len())];
        if to != from {
            return (from, to);
        }
    }
}

/// Adds the corpus graph through GraphMemory's own write path: one entity per four
/// edges, edges between random entity pairs. Returns the entity ids and the number
/// of distinct relationships, since a repeated pair strengthens the existing edge.
fn populate_graph(graph: &GraphMemory, edge_count: usize) -> Result<(Vec<Uuid>, usize)> {
    let mut rng = SplitMix64::new(GRAPH_SEED ^ edge_count as u64);
    let entity_count = (edge_count / EDGES_PER_ENTITY).max(2);
    let entities = (0..entity_count)
        .map(|ordinal| graph.add_entity(entity_node(ordinal)))
        .collect::<Result<Vec<Uuid>>>()?;
    let mut relationships = HashSet::with_capacity(edge_count);
    for _ in 0..edge_count {
        let (from, to) = distinct_pair(&mut rng, &entities);
        let relation = corpus_relation(rng.below(5));
        let strength = 0.1 + 0.8 * rng.unit();
        relationships.insert(graph.add_relationship(edge(from, to, relation, strength))?);
    }
    Ok((entities, relationships.len()))
}

/// The graph each engine runs in production: GraphMemory over the engine's column
/// store, opened as `open_user_graph_memory` opens it.
fn build_graph(engine: Engine, dir: &Path, edge_count: usize) -> Result<(GraphMemory, Vec<Uuid>)> {
    let (graph, entities, relationships) = match engine {
        #[cfg(feature = "rocksdb")]
        Engine::RocksDb => {
            let graph = veld::storage::RocksDbGraphStore::open(dir, None)?.into_inner();
            let (entities, relationships) = populate_graph(&graph, edge_count)?;
            (graph, entities, relationships)
        }
        #[cfg(feature = "storage-redb")]
        Engine::Redb => staged_redb_graph(dir, edge_count)?,
    };
    let stats = graph.get_stats()?;
    ensure!(
        stats.entity_count == entities.len() && stats.relationship_count == relationships,
        "{} graph holds {} entities and {} relationships; {} and {} were written",
        engine.name(),
        stats.entity_count,
        stats.relationship_count,
        entities.len(),
        relationships
    );
    Ok((graph, entities))
}

/// Each GraphMemory write is its own redb commit, four or more per edge, so the graph
/// is built in memory through the same GraphMemory calls and its rows are copied
/// into the redb graph database in large atomic batches before it is opened.
#[cfg(feature = "storage-redb")]
fn staged_redb_graph(dir: &Path, edge_count: usize) -> Result<(GraphMemory, Vec<Uuid>, usize)> {
    use veld::storage::redb::{open_graph_database, RedbColumnStore};

    let staging = Arc::new(staging::StagedColumnStore::default());
    let (entities, relationships) = {
        let staged = GraphMemory::with_store(staging.clone())?;
        populate_graph(&staged, edge_count)?
    };
    let database = open_graph_database(dir)?;
    staging.copy_into(
        &RedbColumnStore::new(Arc::clone(&database)),
        GRAPH_LOAD_BATCH,
    )?;
    let graph = GraphMemory::with_store(Arc::new(RedbColumnStore::new(database)))?;
    Ok((graph, entities, relationships))
}

#[cfg(feature = "storage-redb")]
mod staging {
    use std::collections::BTreeMap;

    use anyhow::{ensure, Result};
    use parking_lot::RwLock;

    use veld::config::StorageBackend;
    use veld::storage::{BatchOp, ColumnStore, WriteBatch};

    /// The family a batch op without one addresses, as redb names it.
    const DEFAULT_FAMILY: &str = "default";

    type Family = BTreeMap<Vec<u8>, Vec<u8>>;

    /// An in-memory [`ColumnStore`] whose rows are bulk-copied into a redb database.
    #[derive(Default)]
    pub struct StagedColumnStore {
        families: RwLock<BTreeMap<String, Family>>,
    }

    impl StagedColumnStore {
        /// Copies every family and row into `target`, `rows_per_batch` rows per
        /// atomic batch.
        pub fn copy_into(&self, target: &dyn ColumnStore, rows_per_batch: usize) -> Result<usize> {
            let families = self.families.read();
            for name in families.keys() {
                target.ensure_cf(name)?;
            }
            let mut copied = 0usize;
            let mut batch = WriteBatch::new();
            for (name, rows) in families.iter() {
                for (key, value) in rows {
                    batch.put_cf(name.clone(), key.clone(), value.clone());
                    if batch.len() >= rows_per_batch {
                        target.write_batch(&batch)?;
                        copied += batch.len();
                        batch = WriteBatch::new();
                    }
                }
            }
            if !batch.is_empty() {
                target.write_batch(&batch)?;
                copied += batch.len();
            }
            Ok(copied)
        }
    }

    impl ColumnStore for StagedColumnStore {
        fn backend(&self) -> StorageBackend {
            StorageBackend::Redb
        }

        fn get_cf(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
            Ok(self
                .families
                .read()
                .get(cf)
                .and_then(|family| family.get(key))
                .cloned())
        }

        fn put_cf(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
            self.families
                .write()
                .entry(cf.to_string())
                .or_default()
                .insert(key.to_vec(), value.to_vec());
            Ok(())
        }

        fn delete_cf(&self, cf: &str, key: &[u8]) -> Result<()> {
            if let Some(family) = self.families.write().get_mut(cf) {
                family.remove(key);
            }
            Ok(())
        }

        fn scan_cf(
            &self,
            cf: &str,
            prefix: Option<&[u8]>,
            limit: Option<usize>,
        ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
            let families = self.families.read();
            let Some(family) = families.get(cf) else {
                return Ok(Vec::new());
            };
            let start = prefix.unwrap_or_default().to_vec();
            Ok(family
                .range(start..)
                .take_while(|(key, _)| prefix.is_none_or(|p| key.starts_with(p)))
                .take(limit.unwrap_or(usize::MAX))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect())
        }

        fn flush_cf(&self, _cf: &str) -> Result<()> {
            Ok(())
        }

        fn ensure_cf(&self, cf: &str) -> Result<()> {
            self.families.write().entry(cf.to_string()).or_default();
            Ok(())
        }

        fn write_batch(&self, batch: &WriteBatch) -> Result<()> {
            let mut families = self.families.write();
            for name in batch.column_families() {
                ensure!(
                    families.contains_key(name),
                    "write_batch names family '{name}', which was never declared"
                );
            }
            for op in batch.ops() {
                match op {
                    BatchOp::Put { cf, key, value } => {
                        families
                            .entry(cf.clone().unwrap_or_else(|| DEFAULT_FAMILY.to_string()))
                            .or_default()
                            .insert(key.clone(), value.clone());
                    }
                    BatchOp::Delete { cf, key } => {
                        if let Some(family) =
                            families.get_mut(cf.as_deref().unwrap_or(DEFAULT_FAMILY))
                        {
                            family.remove(key);
                        }
                    }
                }
            }
            Ok(())
        }
    }
}

/// Sampling scaled to the corpus, so the 100 000 tier finishes in minutes while the
/// whole-corpus reads still collect ten samples.
fn configure_for_corpus(group: &mut BenchmarkGroup<'_, WallTime>, corpus_size: usize) {
    if corpus_size >= 100_000 {
        group
            .sample_size(10)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(5))
            .sampling_mode(SamplingMode::Flat);
    } else if corpus_size >= 10_000 {
        group
            .sample_size(20)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(3))
            .sampling_mode(SamplingMode::Auto);
    } else {
        group
            .sample_size(30)
            .warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_secs(3))
            .sampling_mode(SamplingMode::Auto);
    }
}

/// One benchmark per engine, record mode and corpus size, over memory records.
fn bench_records<F>(
    c: &mut Criterion,
    operation: &str,
    fixtures: &[EngineFixture],
    elements: impl Fn(&EngineFixture) -> u64,
    mut run: F,
) where
    F: FnMut(&mut Bencher<'_, WallTime>, &EngineFixture, &dyn PrimaryMemoryStore),
{
    let mut group = c.benchmark_group(operation);
    for fixture in fixtures {
        for record in &fixture.records {
            configure_for_corpus(&mut group, fixture.corpus_size);
            group.throughput(Throughput::Elements(elements(fixture)));
            let id = BenchmarkId::new(
                format!("{}-{}", fixture.engine.name(), record.mode.name()),
                fixture.corpus_size,
            );
            group.bench_function(id, |b| run(b, fixture, record.store.as_ref()));
        }
    }
    group.finish();
}

/// One benchmark per engine and corpus size, over the graph, key-value and batch
/// fixtures.
fn bench_services<F>(
    c: &mut Criterion,
    operation: &str,
    fixtures: &[EngineFixture],
    elements: u64,
    mut run: F,
) where
    F: FnMut(&mut Bencher<'_, WallTime>, &EngineFixture),
{
    let mut group = c.benchmark_group(operation);
    for fixture in fixtures {
        configure_for_corpus(&mut group, fixture.corpus_size);
        group.throughput(Throughput::Elements(elements));
        let id = BenchmarkId::new(fixture.engine.name(), fixture.corpus_size);
        group.bench_function(id, |b| run(b, fixture));
    }
    group.finish();
}

fn bench_get_memory(c: &mut Criterion, fixtures: &[EngineFixture]) {
    bench_records(
        c,
        "storage_get_memory",
        fixtures,
        |_| 1,
        |b, fixture, store| {
            let mut rng = SplitMix64::new(READ_SEED);
            b.iter(|| {
                let id = &fixture.ids[rng.below(fixture.ids.len())];
                black_box(
                    store
                        .get_memory(id)
                        .expect("get_memory failed")
                        .expect("a corpus memory was missing"),
                )
            });
        },
    );
}

fn bench_get_all_ids(c: &mut Criterion, fixtures: &[EngineFixture]) {
    bench_records(
        c,
        "storage_get_all_ids",
        fixtures,
        |fixture| fixture.corpus_size as u64,
        |b, _, store| b.iter(|| black_box(store.get_all_ids().expect("get_all_ids failed"))),
    );
}

fn bench_get_all(c: &mut Criterion, fixtures: &[EngineFixture]) {
    bench_records(
        c,
        "storage_get_all",
        fixtures,
        |fixture| fixture.corpus_size as u64,
        |b, _, store| b.iter(|| black_box(store.get_all().expect("get_all failed"))),
    );
}

fn bench_search_by_episode(c: &mut Criterion, fixtures: &[EngineFixture]) {
    let target = storage_corpus::target_episode();
    bench_records(
        c,
        "storage_search_by_episode",
        fixtures,
        |fixture| (fixture.corpus_size / 100).max(1) as u64,
        |b, _, store| {
            b.iter(|| {
                black_box(
                    store
                        .search(SearchCriteria::ByEpisode(target.clone()))
                        .expect("search failed"),
                )
            })
        },
    );
}

fn bench_put_memory(c: &mut Criterion, fixtures: &[EngineFixture]) {
    bench_records(
        c,
        "storage_put_memory",
        fixtures,
        |_| 1,
        |b, fixture, store| {
            let mut generator = CorpusGenerator::new(WRITE_SEED);
            let mut ordinal = fixture.corpus_size;
            b.iter_custom(|iters| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iters {
                    let memory = generator.memory(ordinal);
                    ordinal += 1;
                    let start = Instant::now();
                    store.put_memory(&memory).expect("put_memory failed");
                    elapsed += start.elapsed();
                    store
                        .delete_memory(&memory.id)
                        .expect("removing the benchmark's insert failed");
                }
                elapsed
            });
        },
    );
}

fn bench_delete_memory(c: &mut Criterion, fixtures: &[EngineFixture]) {
    bench_records(
        c,
        "storage_delete_memory",
        fixtures,
        |_| 1,
        |b, fixture, store| {
            let mut generator = CorpusGenerator::new(WRITE_SEED);
            let mut ordinal = fixture.corpus_size;
            b.iter_custom(|iters| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iters {
                    let memory = generator.memory(ordinal);
                    ordinal += 1;
                    store
                        .put_memory(&memory)
                        .expect("inserting the memory to delete failed");
                    let start = Instant::now();
                    store
                        .delete_memory(&memory.id)
                        .expect("delete_memory failed");
                    elapsed += start.elapsed();
                }
                elapsed
            });
        },
    );
}

fn batch_key(window: u64, slot: usize) -> Vec<u8> {
    format!("bench:batch:{window:012}:{slot:03}").into_bytes()
}

/// Fifty puts into `window` interleaved with fifty deletes of the previous window's
/// keys, so the family stays the same size however many batches run.
fn mixed_batch(rng: &mut SplitMix64, window: u64) -> WriteBatch {
    let mut batch = WriteBatch::new();
    for slot in 0..WRITE_BATCH_OPS / 2 {
        batch.put_cf(
            WRITE_BATCH_FAMILY,
            batch_key(window, slot),
            random_bytes(rng, KV_VALUE_BYTES),
        );
        batch.delete_cf(WRITE_BATCH_FAMILY, batch_key(window - 1, slot));
    }
    batch
}

fn bench_write_batch(c: &mut Criterion, fixtures: &[EngineFixture]) {
    bench_services(
        c,
        "storage_write_batch",
        fixtures,
        WRITE_BATCH_OPS as u64,
        |b, fixture| {
            let mut rng = SplitMix64::new(WRITE_SEED);
            let mut window = 1u64;
            b.iter_custom(|iters| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iters {
                    let batch = mixed_batch(&mut rng, window);
                    window += 1;
                    let start = Instant::now();
                    fixture
                        .batch_writer
                        .write_batch(&batch)
                        .expect("write_batch failed");
                    elapsed += start.elapsed();
                }
                elapsed
            });
        },
    );
}

fn bench_kv_get(c: &mut Criterion, fixtures: &[EngineFixture]) {
    bench_services(c, "storage_kv_get", fixtures, 1, |b, fixture| {
        let mut rng = SplitMix64::new(READ_SEED);
        b.iter(|| {
            let key = &fixture.kv_keys[rng.below(fixture.kv_keys.len())];
            black_box(
                fixture
                    .key_values
                    .get_bytes(key)
                    .expect("get_bytes failed")
                    .expect("a fixture key was missing"),
            )
        });
    });
}

fn bench_kv_put(c: &mut Criterion, fixtures: &[EngineFixture]) {
    bench_services(c, "storage_kv_put", fixtures, 1, |b, fixture| {
        let mut rng = SplitMix64::new(WRITE_SEED);
        let values: Vec<Vec<u8>> = (0..KV_VALUE_POOL)
            .map(|_| random_bytes(&mut rng, KV_VALUE_BYTES))
            .collect();
        b.iter(|| {
            let key = &fixture.kv_keys[rng.below(fixture.kv_keys.len())];
            let value = &values[rng.below(values.len())];
            fixture
                .key_values
                .put_bytes(key, value)
                .expect("put_bytes failed");
        });
    });
}

fn bench_graph_neighbours(c: &mut Criterion, fixtures: &[EngineFixture]) {
    bench_services(c, "storage_graph_neighbours", fixtures, 1, |b, fixture| {
        let mut rng = SplitMix64::new(READ_SEED);
        b.iter(|| {
            let entity = &fixture.entities[rng.below(fixture.entities.len())];
            black_box(
                fixture
                    .graph
                    .get_entity_relationships(entity)
                    .expect("get_entity_relationships failed"),
            )
        });
    });
}

/// Inserts a `CoRetrieved` edge, a relation the corpus never holds, so every insert
/// creates an edge rather than strengthening one, and removes it untimed.
fn bench_graph_put_edge(c: &mut Criterion, fixtures: &[EngineFixture]) {
    bench_services(c, "storage_graph_put_edge", fixtures, 1, |b, fixture| {
        let mut rng = SplitMix64::new(WRITE_SEED);
        b.iter_custom(|iters| {
            let mut elapsed = Duration::ZERO;
            for _ in 0..iters {
                let (from, to) = distinct_pair(&mut rng, &fixture.entities);
                let candidate = edge(from, to, RelationType::CoRetrieved, 0.5);
                let start = Instant::now();
                let inserted = fixture
                    .graph
                    .add_relationship(candidate)
                    .expect("add_relationship failed");
                elapsed += start.elapsed();
                fixture
                    .graph
                    .delete_relationship(&inserted)
                    .expect("removing the benchmark's edge failed");
            }
            elapsed
        });
    });
}

fn storage_engine_benchmarks(c: &mut Criterion) {
    let fixtures = build_fixtures()
        .unwrap_or_else(|e| panic!("building the storage-engine fixtures failed: {e:#}"));

    bench_get_memory(c, &fixtures);
    bench_get_all_ids(c, &fixtures);
    bench_get_all(c, &fixtures);
    bench_search_by_episode(c, &fixtures);
    bench_kv_get(c, &fixtures);
    bench_graph_neighbours(c, &fixtures);

    bench_put_memory(c, &fixtures);
    bench_delete_memory(c, &fixtures);
    bench_write_batch(c, &fixtures);
    bench_kv_put(c, &fixtures);
    bench_graph_put_edge(c, &fixtures);
}

criterion_group!(benches, storage_engine_benchmarks);
criterion_main!(benches);
