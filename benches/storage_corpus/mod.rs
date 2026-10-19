//! The corpus and primary-store fixtures the storage-engine benchmarks share.
//!
//! One deterministic corpus per size is generated once and loaded into every engine
//! the build carries, so the engines are measured over byte-identical records.
//!
//! Every store opens at the process's [`StorageTuning`], read from the environment
//! exactly as the server reads it, so the durability class a benchmark measures is the
//! one the server runs at:
//! - `VELD_STORAGE_DURABILITY` — `eventual` (the default: RocksDB writes its WAL without a
//!   sync, redb commits `Durability::Eventual`) or `immediate` (RocksDB syncs every write,
//!   redb commits `Durability::Immediate`)
//! - `VELD_REDB_CACHE_BYTES`, `VELD_ROCKSDB_WRITE_BUFFER_BYTES`,
//!   `VELD_ROCKSDB_BLOCK_CACHE_BYTES` — the caches and memtables, as on the server
//!
//! Loading follows each engine's own write path where that path is cheap enough to
//! drive 100 000 times. RocksDB takes every record through `put_memory`. redb commits
//! each `put_memory` as its own transaction, and each commit reaches the disk
//! (`F_FULLFSYNC` under `immediate`, `F_BARRIERFSYNC` under `eventual` on macOS), which
//! would make a 100 000-record fixture take longer to build than the benchmarks take to
//! run. redb is therefore loaded the way `veld migrate` loads it: the encoded records
//! land in the `memories` table through the store's atomic `write_batch`, and reopening
//! the store derives its secondary indexes from them. `verify_corpus` then checks the
//! loaded store through the public read API, so a change to the record format or the
//! index derivation fails here instead of producing numbers over a broken store.

use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "storage-redb")]
use anyhow::anyhow;
use anyhow::{ensure, Context, Result};
use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

use veld::config::StorageTuning;
use veld::memory::context::ContextBuilder;
use veld::memory::storage::SearchCriteria;
use veld::memory::types::{ContextId, EpisodeContext, ProjectContext, RichContext};
use veld::memory::{Experience, ExperienceType, Memory, MemoryId};
#[cfg(feature = "storage-redb")]
use veld::storage::redb::RedbPrimaryMemoryStore;
use veld::storage::PrimaryMemoryStore;
#[cfg(feature = "rocksdb")]
use veld::storage::RocksDbPrimaryMemoryStore;

/// Comma-separated engine names that narrow a run to some of the engines built in.
const ENGINES_ENV: &str = "VELD_BENCH_ENGINES";

/// The passphrase variable the keystore reads when a store opens.
const PASSPHRASE_ENV: &str = "VELD_MASTER_PASSPHRASE";
const BENCH_PASSPHRASE: &str = "storage-engine-benchmarks-passphrase";

/// MiniLM's width, the dimension the default embedder writes.
pub const EMBEDDING_DIMENSION: usize = 384;

/// Episodes the corpus is spread over; each holds 1% of the records.
const EPISODE_COUNT: usize = 100;
const CONTENT_CHARS: usize = 300;
const AGE_SPAN_SECONDS: usize = 90 * 24 * 60 * 60;
const VERIFY_SAMPLES: usize = 16;

/// The table `put_memory` writes on redb, and the one portable dumps carry.
#[cfg(feature = "storage-redb")]
const REDB_MEMORIES_TABLE: &str = "memories";
#[cfg(feature = "storage-redb")]
const REDB_LOAD_BATCH: usize = 5_000;

const WORDS: &[&str] = &[
    "the",
    "index",
    "rebuild",
    "finished",
    "after",
    "the",
    "consolidation",
    "pass",
    "and",
    "memory",
    "graph",
    "edges",
    "were",
    "strengthened",
    "while",
    "recall",
    "returned",
    "ranked",
    "results",
    "from",
    "storage",
    "for",
    "session",
    "episode",
    "context",
    "user",
    "asked",
    "about",
    "latency",
    "budget",
    "vector",
    "search",
    "keyword",
    "fusion",
    "tier",
    "promotion",
    "decay",
    "schedule",
    "backup",
    "restore",
    "keystore",
    "rotation",
    "agent",
    "noted",
    "that",
    "deployment",
    "worker",
    "queue",
    "drained",
    "before",
    "shutdown",
    "replay",
    "journal",
    "segment",
    "compaction",
    "threshold",
    "embedding",
    "dimension",
    "migration",
    "verified",
    "digest",
    "frame",
    "count",
    "release",
];

const TAGS: &[&str] = &[
    "storage",
    "retrieval",
    "consolidation",
    "ingest",
    "auth",
    "release",
    "decision",
    "bug",
    "preference",
    "invariant",
];

const ENTITIES: &[&str] = &[
    "RocksDB",
    "redb",
    "Vamana",
    "Tantivy",
    "MiniLM",
    "Hebbian graph",
    "intent log",
    "keystore",
    "BM25",
    "Zenoh",
    "LM Studio",
    "Claude Code",
];

/// The engines this build carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    #[cfg(feature = "rocksdb")]
    RocksDb,
    #[cfg(feature = "storage-redb")]
    Redb,
}

impl Engine {
    const BUILT: &'static [Engine] = &[
        #[cfg(feature = "rocksdb")]
        Engine::RocksDb,
        #[cfg(feature = "storage-redb")]
        Engine::Redb,
    ];

    pub fn name(self) -> &'static str {
        match self {
            #[cfg(feature = "rocksdb")]
            Engine::RocksDb => "rocksdb",
            #[cfg(feature = "storage-redb")]
            Engine::Redb => "redb",
        }
    }

    /// Every engine the build carries, narrowed by `VELD_BENCH_ENGINES` when set.
    pub fn selected() -> Vec<Engine> {
        let Ok(requested) = std::env::var(ENGINES_ENV) else {
            return Self::BUILT.to_vec();
        };
        requested
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(|name| {
                Self::BUILT
                    .iter()
                    .copied()
                    .find(|engine| engine.name() == name)
                    .unwrap_or_else(|| {
                        let built: Vec<&str> = Self::BUILT.iter().map(|e| e.name()).collect();
                        panic!("{ENGINES_ENV} names {name:?}; this build carries {built:?}")
                    })
            })
            .collect()
    }
}

/// Corpus sizes from the comma-separated `variable`, or `defaults` when it is unset.
/// Underscores are accepted as digit separators.
pub fn selected_sizes(variable: &str, defaults: &[usize]) -> Vec<usize> {
    let Ok(requested) = std::env::var(variable) else {
        return defaults.to_vec();
    };
    requested
        .split(',')
        .map(|size| size.trim().replace('_', ""))
        .filter(|size| !size.is_empty())
        .map(|size| match size.parse::<usize>() {
            Ok(parsed) if parsed > 0 => parsed,
            _ => panic!("{variable} holds {size:?}; expected positive integers"),
        })
        .collect()
}

/// SplitMix64: small, fast and the same sequence on every platform.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..bound`. The modulo bias is below 2^-40 for every bound used here.
    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// A value in `[0, 1)` with 24 bits of precision.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn uuid(&mut self) -> Uuid {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&self.next_u64().to_le_bytes());
        bytes[8..].copy_from_slice(&self.next_u64().to_le_bytes());
        uuid::Builder::from_random_bytes(bytes).into_uuid()
    }
}

/// The episode the corpus holds 1% of its records in, the one `search` is asked for.
pub fn target_episode() -> String {
    episode_id(0)
}

fn episode_id(episode: usize) -> String {
    format!("episode-{episode:03}")
}

fn episode_of(memory: &Memory) -> Option<&str> {
    memory
        .experience
        .context
        .as_ref()
        .and_then(|context| context.episode.episode_id.as_deref())
}

fn experience_type(choice: usize) -> ExperienceType {
    match choice % 6 {
        0 => ExperienceType::Observation,
        1 => ExperienceType::Conversation,
        2 => ExperienceType::Decision,
        3 => ExperienceType::Learning,
        4 => ExperienceType::Discovery,
        _ => ExperienceType::Task,
    }
}

/// Builds memories shaped like the ones `remember` stores: about 300 characters of
/// content, a 384-float unit embedding, three tags, two entities and an episode.
pub struct CorpusGenerator {
    rng: SplitMix64,
    context: RichContext,
    epoch: DateTime<Utc>,
}

impl CorpusGenerator {
    pub fn new(seed: u64) -> Self {
        let context = ContextBuilder::new()
            .with_project(ProjectContext {
                name: Some("veld".to_string()),
                ..Default::default()
            })
            .build();
        Self {
            rng: SplitMix64::new(seed),
            context,
            epoch: Utc::now(),
        }
    }

    pub fn corpus(&mut self, size: usize) -> Vec<Memory> {
        (0..size).map(|ordinal| self.memory(ordinal)).collect()
    }

    /// The memory at `ordinal`; its episode is `ordinal % 100`.
    pub fn memory(&mut self, ordinal: usize) -> Memory {
        let mut context = self.context.clone();
        context.id = ContextId(self.rng.uuid());
        context.episode = EpisodeContext {
            episode_id: Some(episode_id(ordinal % EPISODE_COUNT)),
            sequence_number: Some(u32::try_from(ordinal / EPISODE_COUNT).unwrap_or(u32::MAX)),
            ..Default::default()
        };
        let experience = Experience {
            experience_type: experience_type(self.rng.below(6)),
            content: self.content(),
            context: Some(context),
            entities: self.distinct(ENTITIES, 2),
            tags: self.distinct(TAGS, 3),
            embeddings: Some(self.embedding()),
            ..Default::default()
        };
        let importance = 0.2 + 0.7 * self.rng.unit();
        let age = Duration::seconds(self.rng.below(AGE_SPAN_SECONDS) as i64);
        Memory::new(
            MemoryId(self.rng.uuid()),
            experience,
            importance,
            None,
            None,
            None,
            Some(self.epoch - age),
        )
    }

    fn content(&mut self) -> String {
        let mut text = String::with_capacity(CONTENT_CHARS + 16);
        while text.len() < CONTENT_CHARS {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(WORDS[self.rng.below(WORDS.len())]);
        }
        text.push('.');
        text
    }

    fn distinct(&mut self, pool: &[&str], count: usize) -> Vec<String> {
        let mut chosen: Vec<String> = Vec::with_capacity(count);
        while chosen.len() < count {
            let candidate = pool[self.rng.below(pool.len())];
            if !chosen.iter().any(|existing| existing == candidate) {
                chosen.push(candidate.to_string());
            }
        }
        chosen
    }

    fn embedding(&mut self) -> Vec<f32> {
        let mut vector: Vec<f32> = (0..EMBEDDING_DIMENSION)
            .map(|_| self.rng.unit() * 2.0 - 1.0)
            .collect();
        let norm = vector
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(f32::EPSILON);
        vector.iter_mut().for_each(|x| *x /= norm);
        vector
    }
}

/// Logs the tuning every store in this run opens at: a result compares only with one
/// measured at the same durability class and cache sizes.
pub fn announce_tuning(bench: &str) {
    let tuning = StorageTuning::process();
    let block_cache = tuning
        .rocksdb_block_cache_bytes
        .map_or_else(|| "built-in".to_string(), |bytes| bytes.to_string());
    eprintln!(
        "{bench}: VELD_STORAGE_DURABILITY={} VELD_REDB_CACHE_BYTES={} \
         VELD_ROCKSDB_WRITE_BUFFER_BYTES={} VELD_ROCKSDB_BLOCK_CACHE_BYTES={block_cache}",
        tuning.durability, tuning.redb_cache_bytes, tuning.rocksdb_write_buffer_bytes,
    );
}

/// A primary store opened by its concrete type, so engine-specific handles (the
/// batch writer's database, MemorySystem's backup handle) stay reachable.
pub enum OpenStore {
    #[cfg(feature = "rocksdb")]
    RocksDb(Arc<RocksDbPrimaryMemoryStore>),
    #[cfg(feature = "storage-redb")]
    Redb(Arc<RedbPrimaryMemoryStore>),
}

impl OpenStore {
    /// Opens the store at `dir` exactly as the server opens a tenant's store, at the
    /// process's [`StorageTuning`].
    pub fn open(engine: Engine, dir: &Path) -> Result<Self> {
        let tuning = StorageTuning::process();
        Ok(match engine {
            #[cfg(feature = "rocksdb")]
            Engine::RocksDb => OpenStore::RocksDb(Arc::new(
                RocksDbPrimaryMemoryStore::open_with_tuning(dir, None, tuning)
                    .with_context(|| format!("opening rocksdb at {}", dir.display()))?,
            )),
            #[cfg(feature = "storage-redb")]
            Engine::Redb => OpenStore::Redb(Arc::new(
                RedbPrimaryMemoryStore::open_with_tuning(dir, tuning)
                    .with_context(|| format!("opening redb at {}", dir.display()))?,
            )),
        })
    }

    pub fn primary(&self) -> Arc<dyn PrimaryMemoryStore> {
        match self {
            #[cfg(feature = "rocksdb")]
            OpenStore::RocksDb(store) => Arc::clone(store) as Arc<dyn PrimaryMemoryStore>,
            #[cfg(feature = "storage-redb")]
            OpenStore::Redb(store) => Arc::clone(store) as Arc<dyn PrimaryMemoryStore>,
        }
    }

    fn encrypts_records(&self) -> bool {
        match self {
            #[cfg(feature = "rocksdb")]
            OpenStore::RocksDb(store) => store.inner().encrypts_records(),
            #[cfg(feature = "storage-redb")]
            OpenStore::Redb(store) => store.encrypts_records(),
        }
    }
}

/// Runs `open` with the benchmark passphrase set when `sealed`, and with no
/// passphrase otherwise, so a passphrase in the caller's shell cannot seal a store
/// meant to be plaintext. The keystore is unsealed at open, so the variable is
/// cleared again as soon as `open` returns.
fn with_passphrase<T>(sealed: bool, open: impl FnOnce() -> Result<T>) -> Result<T> {
    if sealed {
        std::env::set_var(PASSPHRASE_ENV, BENCH_PASSPHRASE);
    } else {
        std::env::remove_var(PASSPHRASE_ENV);
    }
    let opened = open();
    std::env::remove_var(PASSPHRASE_ENV);
    opened
}

fn open_store(engine: Engine, dir: &Path, sealed: bool) -> Result<OpenStore> {
    let store = with_passphrase(sealed, || OpenStore::open(engine, dir))?;
    ensure!(
        store.encrypts_records() == sealed,
        "{} store at {} opened with record encryption {}, the benchmark asked for {}",
        engine.name(),
        dir.display(),
        store.encrypts_records(),
        sealed
    );
    Ok(store)
}

/// Creates a store at `dir`, sealed under encryption-v2 when `sealed`, loads
/// `memories` into it and verifies what it reads back.
pub fn load_corpus(
    engine: Engine,
    dir: &Path,
    sealed: bool,
    memories: &[Memory],
) -> Result<OpenStore> {
    let store = match open_store(engine, dir, sealed)? {
        #[cfg(feature = "rocksdb")]
        OpenStore::RocksDb(store) => {
            for memory in memories {
                store.put_memory(memory)?;
            }
            store.flush()?;
            OpenStore::RocksDb(store)
        }
        #[cfg(feature = "storage-redb")]
        OpenStore::Redb(store) => {
            bulk_load_redb(&store, dir, sealed, memories)?;
            drop(store);
            open_store(engine, dir, sealed)?
        }
    };
    verify_corpus(store.primary().as_ref(), memories).with_context(|| {
        format!(
            "verifying the {} corpus at {}",
            engine.name(),
            dir.display()
        )
    })?;
    Ok(store)
}

/// Writes `memories` into the `memories` table in batches, encoded as `put_memory`
/// encodes them: bincode, then the keystore's record envelope bound to the memory id
/// when the store is sealed.
#[cfg(feature = "storage-redb")]
fn bulk_load_redb(
    store: &RedbPrimaryMemoryStore,
    dir: &Path,
    sealed: bool,
    memories: &[Memory],
) -> Result<()> {
    let cryptors = if sealed {
        let keystore = veld::keystore::store_keystore_path(dir, veld::config::StorageBackend::Redb);
        let keys = with_passphrase(true, || veld::keystore::open_record_keys(&keystore))?
            .with_context(|| {
                format!("no keystore at {} after a sealed open", keystore.display())
            })?;
        Some(keys.cryptors)
    } else {
        None
    };
    for chunk in memories.chunks(REDB_LOAD_BATCH) {
        let mut batch = veld::storage::WriteBatch::new();
        for memory in chunk {
            let encoded = bincode::serde::encode_to_vec(memory, bincode::config::standard())
                .map_err(|e| anyhow!("encoding memory {}: {e}", memory.id.0))?;
            let record = match &cryptors {
                Some(cryptors) => cryptors
                    .active()
                    .encrypt_record(&encoded, memory.id.0.as_bytes())?,
                None => encoded,
            };
            batch.put_cf(REDB_MEMORIES_TABLE, memory.id.0.as_bytes().to_vec(), record);
        }
        store.write_batch(&batch)?;
    }
    Ok(())
}

/// The loaded store holds every memory, answers the episode search from its indexes,
/// and returns the content that was written.
fn verify_corpus(store: &dyn PrimaryMemoryStore, memories: &[Memory]) -> Result<()> {
    let stored = store.get_all_ids()?.len();
    ensure!(
        stored == memories.len(),
        "the store holds {stored} memories after loading {}",
        memories.len()
    );

    let target = target_episode();
    let expected = memories
        .iter()
        .filter(|memory| episode_of(memory) == Some(target.as_str()))
        .count();
    let found = store
        .search(SearchCriteria::ByEpisode(target.clone()))?
        .len();
    ensure!(
        found == expected,
        "search for {target} returned {found} memories; the corpus holds {expected}"
    );

    let stride = (memories.len() / VERIFY_SAMPLES).max(1);
    for memory in memories.iter().step_by(stride) {
        let read = store
            .get_memory(&memory.id)?
            .with_context(|| format!("memory {} was not readable", memory.id.0))?;
        ensure!(
            read.experience.content == memory.experience.content,
            "memory {} read back different content",
            memory.id.0
        );
    }
    Ok(())
}
