//! Cold-start cost per storage engine for ADR 0007: how long a populated tenant takes
//! to come back after a restart.
//!
//! Groups, each with benchmark id `{engine}/{corpus size}`:
//! - `cold_start_store_open` — the primary store opened over a populated directory.
//! - `cold_start_memory_system` — everything `MemorySystem` construction does on an
//!   open store: BM25 open, embedder resolution, the corpus-dimension check, the
//!   saved vector index load, the startup orphan check and the state loads.
//! - `cold_start_vector_index_load` — the retrieval engine alone, loading the saved
//!   Vamana index and checking it against the stored vector mappings.
//! - `cold_start_vector_index_rebuild` — the retrieval engine over an index directory
//!   with no saved index and no corpus markers, as on the first boot after
//!   `veld migrate`: a full corpus-dimension pass, a Vamana build from the stored
//!   embeddings, and one vector-mapping write per memory.
//!
//! Each fixture is populated once per engine and size and booted once, which builds
//! and saves its vector index; every timed boot reopens that directory. The embedder
//! is pinned to offline hash-mode MiniLM (384d), so no model file and no network is
//! touched and the stored embeddings are used as they are.
//!
//! A rebuild writes one vector mapping per memory, and redb commits each one as its own
//! transaction, so a 100 000-memory rebuild on redb runs for minutes and Criterion needs
//! eleven of them. The rebuild group therefore runs at 10 000 by default.
//!
//! Every store opens at the process's `StorageTuning`, as the server's do, and the run
//! logs the durability class and cache sizes it measured at.
//!
//! Environment:
//! - `VELD_BENCH_CORPUS_SIZES` — comma-separated corpus sizes (default `10000,100000`)
//! - `VELD_BENCH_REBUILD_SIZES` — sizes the rebuild group runs at (default `10000`)
//! - `VELD_BENCH_ENGINES` — comma-separated engines (default: every engine built in)
//! - `VELD_STORAGE_DURABILITY` — `eventual` (default) or `immediate`, for both engines
//! - `VELD_REDB_CACHE_BYTES`, `VELD_ROCKSDB_WRITE_BUFFER_BYTES`,
//!   `VELD_ROCKSDB_BLOCK_CACHE_BYTES` — cache and memtable sizes, as on the server

mod storage_corpus;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use tempfile::TempDir;

use veld::embeddings::minilm::{EmbeddingConfig, MiniLMEmbedder};
use veld::embeddings::CompetitiveEmbedder;
use veld::memory::hybrid_search::BM25Index;
use veld::memory::retrieval::RetrievalEngine;
use veld::memory::{Memory, MemoryConfig, MemorySystem, BM25_INDEX_DIR};

use storage_corpus::{load_corpus, CorpusGenerator, Engine, OpenStore};

const CORPUS_SIZES_ENV: &str = "VELD_BENCH_CORPUS_SIZES";
const REBUILD_SIZES_ENV: &str = "VELD_BENCH_REBUILD_SIZES";
const DEFAULT_CORPUS_SIZES: &[usize] = &[10_000, 100_000];
const DEFAULT_REBUILD_SIZES: &[usize] = &[10_000];

const CORPUS_SEED: u64 = 0x5EED_B007_0001;

/// The embedder every boot resolves: MiniLM by name, offline, with no model files in
/// reach, so it runs in hash mode at 384 dimensions. The HTTP probe for a secondary
/// embedder is pointed at the discard port, where nothing answers.
fn pin_offline_embedder(empty_model_dir: &Path) {
    std::env::set_var("VELD_EMBEDDER", "minilm");
    std::env::set_var("VELD_OFFLINE", "1");
    std::env::set_var("VELD_ALLOW_SIMPLIFIED_EMBEDDINGS", "1");
    std::env::set_var("VELD_MODEL_PATH", empty_model_dir);
    std::env::set_var("VELD_EMBEDDING_API_URL", "http://127.0.0.1:9");
}

/// The embedder `pin_offline_embedder` makes MemorySystem resolve, for the retrieval
/// engine benchmarks, so their corpus markers agree with the fixture's.
fn retrieval_embedder() -> Result<Arc<CompetitiveEmbedder>> {
    let minilm = MiniLMEmbedder::new_simplified(EmbeddingConfig::from_env())?;
    Ok(Arc::new(CompetitiveEmbedder::primary_only(Arc::new(
        minilm,
    ))))
}

fn memory_config(dir: &Path) -> MemoryConfig {
    MemoryConfig {
        storage_path: dir.to_path_buf(),
        ..Default::default()
    }
}

/// Constructs MemorySystem on `store` the way the server constructs a tenant's.
fn boot_memory_system(store: OpenStore, dir: &Path) -> Result<MemorySystem> {
    let bm25 = Arc::new(BM25Index::new(&dir.join(BM25_INDEX_DIR))?);
    match store {
        #[cfg(feature = "rocksdb")]
        OpenStore::RocksDb(store) => {
            let engine_handle = store.inner().db();
            MemorySystem::with_primary_store_and_bm25_index(
                memory_config(dir),
                store,
                Some(engine_handle),
                bm25,
            )
        }
        #[cfg(feature = "storage-redb")]
        OpenStore::Redb(store) => {
            let database = store.database();
            let mut system = MemorySystem::with_primary_store_and_bm25_index(
                memory_config(dir),
                store,
                None,
                bm25,
            )?;
            system.attach_redb_backup_db(database);
            Ok(system)
        }
    }
}

fn ensure_fully_indexed(system: &MemorySystem, expected: usize, stage: &str) -> Result<()> {
    let stats = system.stats();
    ensure!(
        stats.total_memories == expected && stats.vector_index_count == expected,
        "{stage}: {} memories stored and {} indexed, expected {expected} of each",
        stats.total_memories,
        stats.vector_index_count
    );
    Ok(())
}

fn fill_bm25(dir: &Path, memories: &[Memory]) -> Result<()> {
    let index = BM25Index::new(&dir.join(BM25_INDEX_DIR))?;
    for memory in memories {
        index.upsert(
            &memory.id,
            &memory.experience.content,
            &memory.experience.tags,
            &memory.experience.entities,
        )?;
    }
    index.commit()
}

/// A populated tenant directory: primary store, BM25 index, saved vector index and
/// corpus markers, as a clean shutdown leaves them.
struct BootFixture {
    engine: Engine,
    corpus_size: usize,
    directory: TempDir,
}

impl BootFixture {
    fn build(engine: Engine, memories: &[Memory]) -> Result<Self> {
        let directory = TempDir::new()?;
        let dir = directory.path();
        let corpus_size = memories.len();

        drop(load_corpus(engine, dir, false, memories)?);
        fill_bm25(dir, memories)?;

        let started = Instant::now();
        let system = boot_memory_system(OpenStore::open(engine, dir)?, dir)?;
        // A server open leaves what its index lacks to the backlog drainer's tick; the
        // fixture drains it here, so every timed boot starts from a fully indexed tenant.
        system.drain_embedding_backlog_journaled(usize::MAX)?;
        let first_boot = started.elapsed();
        ensure_fully_indexed(&system, corpus_size, "first boot")?;
        system.flush_storage()?;
        drop(system);

        let system = boot_memory_system(OpenStore::open(engine, dir)?, dir)?;
        ensure_fully_indexed(&system, corpus_size, "boot from the saved index")?;
        drop(system);

        eprintln!(
            "cold_start_benchmarks: {} with {corpus_size} memories — the first boot built \
             and indexed the corpus in {:.1}s",
            engine.name(),
            first_boot.as_secs_f64()
        );
        Ok(Self {
            engine,
            corpus_size,
            directory,
        })
    }

    fn path(&self) -> &Path {
        self.directory.path()
    }

    fn open(&self) -> OpenStore {
        OpenStore::open(self.engine, self.path()).expect("reopening the fixture store failed")
    }
}

fn build_fixtures() -> Result<Vec<BootFixture>> {
    storage_corpus::announce_tuning("cold_start_benchmarks");
    let engines = Engine::selected();
    let sizes = storage_corpus::selected_sizes(CORPUS_SIZES_ENV, DEFAULT_CORPUS_SIZES);
    let mut fixtures = Vec::with_capacity(engines.len() * sizes.len());
    for &corpus_size in &sizes {
        let memories = CorpusGenerator::new(CORPUS_SEED).corpus(corpus_size);
        for &engine in &engines {
            eprintln!(
                "cold_start_benchmarks: building the {} fixture of {corpus_size} memories",
                engine.name()
            );
            let fixture = BootFixture::build(engine, &memories).with_context(|| {
                format!(
                    "building the {} boot fixture of {corpus_size}",
                    engine.name()
                )
            })?;
            fixtures.push(fixture);
        }
    }
    Ok(fixtures)
}

/// Times one `boot` per Criterion iteration. A boot returns only the span it measured,
/// so opening inputs and dropping what it built stay outside the timing.
fn bench_boots(
    c: &mut Criterion,
    group_name: &str,
    fixtures: &[&BootFixture],
    mut boot: impl FnMut(&BootFixture) -> Duration,
) {
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(10)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(10));
    for &fixture in fixtures {
        group.throughput(Throughput::Elements(fixture.corpus_size as u64));
        group.bench_function(
            BenchmarkId::new(fixture.engine.name(), fixture.corpus_size),
            |b| b.iter_custom(|iters| (0..iters).map(|_| boot(fixture)).sum()),
        );
    }
    group.finish();
}

fn bench_store_open(c: &mut Criterion, fixtures: &[&BootFixture]) {
    bench_boots(c, "cold_start_store_open", fixtures, |fixture| {
        let start = Instant::now();
        let store = fixture.open();
        let elapsed = start.elapsed();
        drop(store);
        elapsed
    });
}

fn bench_memory_system(c: &mut Criterion, fixtures: &[&BootFixture]) {
    bench_boots(c, "cold_start_memory_system", fixtures, |fixture| {
        let store = fixture.open();
        let start = Instant::now();
        let system =
            boot_memory_system(store, fixture.path()).expect("MemorySystem construction failed");
        let elapsed = start.elapsed();
        drop(system);
        elapsed
    });
}

fn bench_vector_index_load(
    c: &mut Criterion,
    fixtures: &[&BootFixture],
    embedder: &Arc<CompetitiveEmbedder>,
) {
    bench_boots(c, "cold_start_vector_index_load", fixtures, |fixture| {
        let store = fixture.open();
        let primary = store.primary();
        let embedder = Arc::clone(embedder);
        let start = Instant::now();
        let engine = RetrievalEngine::new(primary, fixture.path(), embedder)
            .expect("retrieval engine construction failed");
        let elapsed = start.elapsed();
        assert_eq!(
            engine.len(),
            fixture.corpus_size,
            "the loaded index does not cover the corpus"
        );
        drop(engine);
        drop(store);
        elapsed
    });
}

fn bench_vector_index_rebuild(
    c: &mut Criterion,
    fixtures: &[&BootFixture],
    embedder: &Arc<CompetitiveEmbedder>,
) {
    bench_boots(c, "cold_start_vector_index_rebuild", fixtures, |fixture| {
        let store = fixture.open();
        let primary = store.primary();
        let embedder = Arc::clone(embedder);
        let index_dir = TempDir::new().expect("creating the rebuild's index directory failed");
        let start = Instant::now();
        let engine = RetrievalEngine::new(primary, index_dir.path(), embedder)
            .expect("retrieval engine rebuild failed");
        let elapsed = start.elapsed();
        assert_eq!(
            engine.len(),
            fixture.corpus_size,
            "the rebuilt index does not cover the corpus"
        );
        drop(engine);
        drop(store);
        elapsed
    });
}

fn cold_start_benchmarks(c: &mut Criterion) {
    let empty_model_dir = TempDir::new().expect("creating the empty model directory failed");
    pin_offline_embedder(empty_model_dir.path());
    let embedder = retrieval_embedder().expect("constructing the hash-mode embedder failed");

    let fixtures = build_fixtures()
        .unwrap_or_else(|e| panic!("building the cold-start fixtures failed: {e:#}"));
    let all: Vec<&BootFixture> = fixtures.iter().collect();

    let rebuild_sizes = storage_corpus::selected_sizes(REBUILD_SIZES_ENV, DEFAULT_REBUILD_SIZES);
    for size in &rebuild_sizes {
        if !fixtures.iter().any(|fixture| fixture.corpus_size == *size) {
            eprintln!(
                "cold_start_benchmarks: {REBUILD_SIZES_ENV} names {size}, which \
                 {CORPUS_SIZES_ENV} does not build; no rebuild runs at that size"
            );
        }
    }
    let rebuilt: Vec<&BootFixture> = fixtures
        .iter()
        .filter(|fixture| rebuild_sizes.contains(&fixture.corpus_size))
        .collect();

    bench_store_open(c, &all);
    bench_memory_system(c, &all);
    bench_vector_index_load(c, &all, &embedder);
    bench_vector_index_rebuild(c, &rebuilt, &embedder);
}

criterion_group!(benches, cold_start_benchmarks);
criterion_main!(benches);
