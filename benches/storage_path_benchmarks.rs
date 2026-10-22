//! The paths above the storage engines that a write or a tenant open runs through, which
//! `storage_engine_benchmarks` (trait operations only) does not reach.
//!
//! Groups:
//! - `journal_append` — one `Remember` frame through `SharedJournal::record_and_apply`, and
//!   a page of 64 through `record_and_apply_batch`, with no projection attached: the frame
//!   encode, the append and the group sync. Benchmark id `single` or `batch-64`.
//! - `bm25_projection_apply` — one `Remember` frame applied to a `Bm25Projection` through
//!   `TypedProjection::apply`, as the journal applies it live: the document is buffered and
//!   the background committer commits on its own cadence. Id `{documents already indexed}`.
//! - `vamana_insert` — one 384-dimension vector added to a built Vamana index. Id
//!   `{vectors in the index}`.
//! - `tenant_open` — a populated tenant opened the way the server opens one: the primary
//!   store, `MemorySystem` over it, and the tenant's graph. Id `{engine}-{graph}/{memories}`,
//!   where `graph` is `saved-counters` (the graph was closed with its counters saved, so the
//!   open reads them) or `counted` (no saved counters, so the open counts every
//!   relationship).
//!
//! The embedder is pinned to offline hash-mode MiniLM (384d), so no model file and no
//! network is touched. Every store opens at the process's `StorageTuning`, and the run logs
//! the durability class and cache sizes it measured at.
//!
//! Environment:
//! - `VELD_BENCH_CORPUS_SIZES` — tenant sizes for `tenant_open` (default `10000`)
//! - `VELD_BENCH_INDEX_SIZES` — index sizes for `bm25_projection_apply` and `vamana_insert`
//!   (default `10000`)
//! - `VELD_BENCH_ENGINES` — comma-separated engines (default: every engine built in)
//! - `VELD_STORAGE_DURABILITY`, `VELD_REDB_CACHE_BYTES`, `VELD_ROCKSDB_WRITE_BUFFER_BYTES`,
//!   `VELD_ROCKSDB_BLOCK_CACHE_BYTES` — as on the server

mod storage_corpus;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use parking_lot::Mutex;
use tempfile::TempDir;
use uuid::Uuid;

use veld::graph_memory::{
    EdgeSource, EdgeTier, EntityLabel, EntityNode, GraphMemory, LtpStatus, PiiClassification,
    RelationType, RelationshipEdge,
};
use veld::intent_log::payload::CURRENT_PAYLOAD_SCHEMA_VERSION;
use veld::intent_log::{
    CheckpointStore, IntentLog, IntentPayload, JournaledWriter, Lsn, SharedJournal, TypedProjection,
};
use veld::memory::hybrid_search::BM25Index;
use veld::memory::{Bm25Projection, Memory, MemoryConfig, MemorySystem, BM25_INDEX_DIR};
use veld::vector_db::vamana::{VamanaConfig, VamanaIndex};

use storage_corpus::{load_corpus, CorpusGenerator, Engine, OpenStore, SplitMix64};

const CORPUS_SIZES_ENV: &str = "VELD_BENCH_CORPUS_SIZES";
const INDEX_SIZES_ENV: &str = "VELD_BENCH_INDEX_SIZES";
const DEFAULT_CORPUS_SIZES: &[usize] = &[10_000];
const DEFAULT_INDEX_SIZES: &[usize] = &[10_000];

const CORPUS_SEED: u64 = 0x5EED_9A7B_0001;
const PAYLOAD_SEED: u64 = 0x5EED_9A7B_0002;
const VECTOR_SEED: u64 = 0x5EED_9A7B_0003;

/// Frames cycled through by the append and apply benchmarks, so neither measures one
/// payload's cache-warm encode.
const PAYLOAD_POOL: usize = 256;
const JOURNAL_BATCH: usize = 64;
/// Graph shape for `tenant_open`: one edge per memory, four edges per entity.
const EDGES_PER_ENTITY: usize = 4;
const GRAPH_LOAD_BATCH: usize = 1_000;

/// The embedder every open resolves: MiniLM by name, offline, with no model files in
/// reach, so it runs in hash mode at 384 dimensions. The HTTP probe for a secondary
/// embedder is pointed at the discard port, where nothing answers.
fn pin_offline_embedder(empty_model_dir: &Path) {
    std::env::set_var("VELD_EMBEDDER", "minilm");
    std::env::set_var("VELD_OFFLINE", "1");
    std::env::set_var("VELD_ALLOW_SIMPLIFIED_EMBEDDINGS", "1");
    std::env::set_var("VELD_MODEL_PATH", empty_model_dir);
    std::env::set_var("VELD_EMBEDDING_API_URL", "http://127.0.0.1:9");
}

fn remember_payload(memory: &Memory) -> Result<IntentPayload> {
    Ok(IntentPayload::Remember {
        user_id: "bench".to_string(),
        memory_id: memory.id.0.to_string(),
        memory_bincode: bincode::serde::encode_to_vec(memory, bincode::config::standard())?,
        schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
    })
}

fn payload_pool() -> Result<Vec<IntentPayload>> {
    CorpusGenerator::new(PAYLOAD_SEED)
        .corpus(PAYLOAD_POOL)
        .iter()
        .map(remember_payload)
        .collect()
}

fn group_settings(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
}

fn bench_journal_append(c: &mut Criterion, payloads: &[IntentPayload]) -> Result<()> {
    let dir = TempDir::new()?;
    let journal = SharedJournal::new(JournaledWriter::new(IntentLog::open(
        dir.path().join("intent.log"),
    )?));
    let mut group = c.benchmark_group("journal_append");
    group_settings(&mut group);

    let mut next = 0usize;
    group.throughput(Throughput::Elements(1));
    group.bench_function("single", |b| {
        b.iter(|| {
            let payload = &payloads[next % payloads.len()];
            next += 1;
            journal
                .record_and_apply(payload)
                .expect("journal append failed")
        })
    });

    let pages: Vec<&[IntentPayload]> = payloads.chunks(JOURNAL_BATCH).collect();
    let mut page = 0usize;
    group.throughput(Throughput::Elements(JOURNAL_BATCH as u64));
    group.bench_function(format!("batch-{JOURNAL_BATCH}"), |b| {
        b.iter(|| {
            let frames = pages[page % pages.len()];
            page += 1;
            journal
                .record_and_apply_batch(frames)
                .expect("journal batch append failed")
        })
    });
    group.finish();
    Ok(())
}

fn bench_bm25_projection_apply(
    c: &mut Criterion,
    payloads: &[IntentPayload],
    sizes: &[usize],
) -> Result<()> {
    let mut group = c.benchmark_group("bm25_projection_apply");
    group_settings(&mut group);
    group.throughput(Throughput::Elements(1));
    for &size in sizes {
        let dir = TempDir::new()?;
        let index = Arc::new(BM25Index::new(&dir.path().join("bm25"))?);
        for memory in CorpusGenerator::new(CORPUS_SEED).corpus(size) {
            index.upsert(
                &memory.id,
                &memory.experience.content,
                &memory.experience.tags,
                &memory.experience.entities,
            )?;
        }
        index.commit()?;
        let checkpoints = Arc::new(Mutex::new(CheckpointStore::open(
            dir.path().join("checkpoints"),
        )?));
        let mut projection = Bm25Projection::new(index, checkpoints);
        let mut lsn = 0u64;
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                let payload = &payloads[lsn as usize % payloads.len()];
                TypedProjection::apply(&mut projection, Lsn(lsn), payload)
                    .expect("BM25 projection apply failed");
                lsn += 1;
            })
        });
        drop(projection);
    }
    group.finish();
    Ok(())
}

fn random_unit_vector(rng: &mut SplitMix64, dimension: usize) -> Vec<f32> {
    let mut vector: Vec<f32> = (0..dimension).map(|_| rng.unit() * 2.0 - 1.0).collect();
    let norm = vector
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    vector.iter_mut().for_each(|x| *x /= norm);
    vector
}

fn bench_vamana_insert(c: &mut Criterion, sizes: &[usize]) -> Result<()> {
    let dimension = storage_corpus::EMBEDDING_DIMENSION;
    let mut group = c.benchmark_group("vamana_insert");
    group_settings(&mut group);
    group.throughput(Throughput::Elements(1));
    for &size in sizes {
        let mut rng = SplitMix64::new(VECTOR_SEED);
        let mut index = VamanaIndex::new(VamanaConfig {
            dimension,
            use_mmap: false,
            ..Default::default()
        })?;
        index.build(
            (0..size)
                .map(|_| random_unit_vector(&mut rng, dimension))
                .collect(),
        )?;
        let pool: Vec<Vec<f32>> = (0..PAYLOAD_POOL)
            .map(|_| random_unit_vector(&mut rng, dimension))
            .collect();
        let mut next = 0usize;
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                let vector = pool[next % pool.len()].clone();
                next += 1;
                index.add_vector(vector).expect("Vamana insert failed")
            })
        });
    }
    group.finish();
    Ok(())
}

fn entity_node(ordinal: usize) -> EntityNode {
    EntityNode {
        uuid: Uuid::new_v4(),
        name: format!("bench-entity-{ordinal}"),
        labels: vec![EntityLabel::Concept],
        created_at: chrono::Utc::now(),
        last_seen_at: chrono::Utc::now(),
        mention_count: 1,
        summary: String::new(),
        attributes: Default::default(),
        name_embedding: None,
        salience: 0.5,
        is_proper_noun: false,
        pii_classification: PiiClassification::default(),
    }
}

fn edge(from: Uuid, to: Uuid) -> RelationshipEdge {
    let now = chrono::Utc::now();
    RelationshipEdge {
        uuid: Uuid::new_v4(),
        from_entity: from,
        to_entity: to,
        relation_type: RelationType::RelatedTo,
        strength: 0.5,
        created_at: now,
        valid_at: now,
        invalidated_at: None,
        source_episode_id: None,
        context: String::new(),
        last_activated: now,
        activation_count: 1,
        ltp_status: LtpStatus::None,
        activation_timestamps: None,
        tier: EdgeTier::L2Episodic,
        created_by: EdgeSource::CoOccurrence,
        entity_confidence: None,
        forward_strength: 0.5,
        backward_strength: 0.5,
        strengthening_sources: Default::default(),
        last_corroborated: None,
    }
}

/// The tenant's graph at `base`, opened as `open_user_graph_memory` opens it.
fn open_graph(engine: Engine, base: &Path) -> Result<GraphMemory> {
    match engine {
        #[cfg(feature = "rocksdb")]
        Engine::RocksDb => Ok(veld::storage::RocksDbGraphStore::open(base, None)?.into_inner()),
        #[cfg(feature = "storage-redb")]
        Engine::Redb => {
            use veld::storage::redb::{open_graph_database, RedbColumnStore};
            let path = base.join("graph");
            std::fs::create_dir_all(&path)?;
            let database = open_graph_database(&path)?;
            let mut graph =
                GraphMemory::with_store(Arc::new(RedbColumnStore::new(Arc::clone(&database))))?;
            graph.attach_redb_backup_db(database);
            Ok(graph)
        }
    }
}

/// A graph of `edges` edges over `edges / EDGES_PER_ENTITY` entities, written through
/// GraphMemory, then closed with its counters saved when `save_counters`.
fn build_graph(engine: Engine, base: &Path, edges: usize, save_counters: bool) -> Result<()> {
    let mut graph = open_graph(engine, base)?;
    let entities: Vec<Uuid> = (0..(edges / EDGES_PER_ENTITY).max(2))
        .map(|ordinal| graph.add_entity(entity_node(ordinal)))
        .collect::<Result<_>>()?;
    let mut rng = SplitMix64::new(CORPUS_SEED);
    let mut pending = Vec::with_capacity(GRAPH_LOAD_BATCH);
    for _ in 0..edges {
        let from = entities[rng.below(entities.len())];
        let mut to = entities[rng.below(entities.len())];
        while to == from {
            to = entities[rng.below(entities.len())];
        }
        pending.push(edge(from, to));
        if pending.len() == GRAPH_LOAD_BATCH {
            graph.add_relationships(std::mem::take(&mut pending))?;
        }
    }
    if !pending.is_empty() {
        graph.add_relationships(pending)?;
    }
    if save_counters {
        graph.persist_counters()?;
    }
    Ok(())
}

/// A populated tenant directory with two graphs beside its store: one closed with its
/// counters saved and one without.
struct TenantFixture {
    engine: Engine,
    memories: usize,
    directory: TempDir,
}

impl TenantFixture {
    fn build(engine: Engine, corpus: &[Memory]) -> Result<Self> {
        let directory = TempDir::new()?;
        let dir = directory.path();
        drop(load_corpus(engine, dir, false, corpus)?);
        let fixture = Self {
            engine,
            memories: corpus.len(),
            directory,
        };
        // The first open builds and saves the vector index, as a tenant's first boot does.
        let system = fixture.open_memory_system()?;
        system.drain_embedding_backlog_journaled(usize::MAX)?;
        let stats = system.stats();
        ensure!(
            stats.total_memories == corpus.len() && stats.vector_index_count == corpus.len(),
            "{}: {} memories stored and {} indexed, expected {}",
            engine.name(),
            stats.total_memories,
            stats.vector_index_count,
            corpus.len()
        );
        system.flush_storage()?;
        drop(system);
        build_graph(engine, &fixture.graph_base(true), corpus.len(), true)?;
        build_graph(engine, &fixture.graph_base(false), corpus.len(), false)?;
        Ok(fixture)
    }

    fn graph_base(&self, saved_counters: bool) -> PathBuf {
        self.directory.path().join(if saved_counters {
            "graph-saved-counters"
        } else {
            "graph-counted"
        })
    }

    /// MemorySystem over the tenant's store, constructed as the server constructs it.
    fn open_memory_system(&self) -> Result<MemorySystem> {
        let dir = self.directory.path();
        let store = OpenStore::open(self.engine, dir)?;
        let bm25 = Arc::new(BM25Index::new(&dir.join(BM25_INDEX_DIR))?);
        MemorySystem::with_primary_store_and_bm25_index(
            MemoryConfig {
                storage_path: dir.to_path_buf(),
                ..Default::default()
            },
            store.primary(),
            None,
            bm25,
        )
    }
}

fn bench_tenant_open(c: &mut Criterion, fixtures: &[TenantFixture]) {
    let mut group = c.benchmark_group("tenant_open");
    group
        .sample_size(10)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(10));
    for fixture in fixtures {
        group.throughput(Throughput::Elements(fixture.memories as u64));
        for (label, saved_counters) in [("saved-counters", true), ("counted", false)] {
            let id = BenchmarkId::new(
                format!("{}-{label}", fixture.engine.name()),
                fixture.memories,
            );
            let graph_base = fixture.graph_base(saved_counters);
            group.bench_function(id, |b| {
                b.iter_custom(|iters| {
                    (0..iters)
                        .map(|_| {
                            let started = Instant::now();
                            let system = fixture
                                .open_memory_system()
                                .expect("MemorySystem construction failed");
                            let graph =
                                open_graph(fixture.engine, &graph_base).expect("graph open failed");
                            let elapsed = started.elapsed();
                            drop(graph);
                            drop(system);
                            elapsed
                        })
                        .sum()
                })
            });
        }
    }
    group.finish();
}

fn storage_path_benchmarks(c: &mut Criterion) {
    let empty_model_dir = TempDir::new().expect("creating the empty model directory failed");
    pin_offline_embedder(empty_model_dir.path());
    storage_corpus::announce_tuning("storage_path_benchmarks");

    let payloads = payload_pool().expect("encoding the frame pool failed");
    let index_sizes = storage_corpus::selected_sizes(INDEX_SIZES_ENV, DEFAULT_INDEX_SIZES);
    bench_journal_append(c, &payloads).expect("journal_append setup failed");
    bench_bm25_projection_apply(c, &payloads, &index_sizes)
        .expect("bm25_projection_apply setup failed");
    bench_vamana_insert(c, &index_sizes).expect("vamana_insert setup failed");

    let mut fixtures = Vec::new();
    for size in storage_corpus::selected_sizes(CORPUS_SIZES_ENV, DEFAULT_CORPUS_SIZES) {
        let corpus = CorpusGenerator::new(CORPUS_SEED).corpus(size);
        for engine in Engine::selected() {
            eprintln!(
                "storage_path_benchmarks: building the {} tenant of {size} memories",
                engine.name()
            );
            fixtures.push(
                TenantFixture::build(engine, &corpus)
                    .with_context(|| format!("building the {} tenant of {size}", engine.name()))
                    .unwrap_or_else(|e| panic!("{e:#}")),
            );
        }
    }
    bench_tenant_open(c, &fixtures);
}

criterion_group!(benches, storage_path_benchmarks);
criterion_main!(benches);
