#![cfg(all(feature = "rocksdb", feature = "storage-redb"))]
//! ADR 0007 gate: "recall-gate scores are equal on both engines over the same corpus and
//! embedder".
//!
//! Each test builds three tenants in one process from the same corpus, user id, embedder and
//! scoring clock, each behind its own `MultiUserMemoryManager` (the production per-tenant wiring,
//! graph and NER included): a RocksDB tenant (the reference), a second RocksDB tenant (the
//! same-engine control) and a redb tenant (the candidate). Every probe runs on the three in turn,
//! and two comparisons are recorded: reference against redb, which is an engine divergence, and
//! reference against the control, which is run-to-run nondeterminism. A probe that differs in both
//! is reported as nondeterministic; the cross-engine comparison is never loosened for it.
//!
//! The embedder is MiniLM in hash mode (`generate_embedding_simplified` in
//! `src/embeddings/minilm.rs`): `VELD_EMBEDDER=minilm` with `VELD_MODEL_PATH` at a directory that
//! holds no model, opted into with `VELD_ALLOW_SIMPLIFIED_EMBEDDINGS=1`. Its vectors are a pure
//! function of the text, so both engines index the same vectors with no model files and no
//! network; `VELD_OFFLINE` stops downloads and the HTTP secondary points at the discard port.
//! `VELD_EVAL_NOW` freezes the scoring clock and `VELD_RECALL_READONLY=1` keeps recall from
//! teaching the store, both as `recall-eval` pins them.
//!
//! Probes compare, per query: the ranked corpus keys, each score, tier and importance, the
//! `SignalAttribution` recall leaves in `last_signal_attributions`, and the `ScoringFeatures`
//! (base score, the additive scoring components, the multiplicative gates, the unified and final
//! scores) the LTR trace store captures for each returned memory. `ScoringSignals` itself is not
//! returned by recall; `ScoringFeatures` is the form in which recall exposes those signals.
//! Numbers agree within 1e-6, everything else exactly.
//!
//! Normalisations, applied identically to both comparisons:
//! 1. Memory ids become corpus keys. `remember` mints every id with `Uuid::new_v4`, so the
//!    tenants hold different ids for the same corpus item.
//! 2. Results whose scores lie within the tolerance of their neighbour, or that both carry no
//!    score, form a tie group compared in corpus-key order. Every ranking stage breaks ties by
//!    memory id (normalisation 1), so the order inside a tie group is the random id order.
//! 3. `recall_by_tags` and `recall_by_date` are read with a limit above the corpus size and
//!    compared as key sets: both engines return index-key order, and the index key ends in the
//!    memory id. A filter-only recall compares as one tie group by normalisation 2, since its hits
//!    carry no score.
//! 4. `proactive_context` responses keep the fields in `PROACTIVE_MEMORY_FIELDS`. Left out are
//!    `latency_ms`, `hours_since_created`, `hours_since_last_accessed` and `intrusion_score`,
//!    read from the clock at response time; `content` and `embedding`, implied by the key; and
//!    `competition_opponent`, a memory id. Surfaced facts drop their `id`.
//! 5. Semantic facts drop `id`, `created_at`, `last_reinforced`, `valid_from`, `valid_until`,
//!    `superseded_by`, `supersedes` and `purged_at`: fact ids are `Uuid::new_v4` and the stamps
//!    are the wall clock of the maintenance pass. Their source memories become sorted corpus keys,
//!    and the facts are compared in their serialised order.
//! 6. Trace rows are matched by memory and their `rank` is dropped; rank is covered by the ranking
//!    under normalisation 2.
//!
//! Own test binary: the tests pin process-global environment variables, so they run one at a time
//! under a lock.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration as StdDuration, Instant};

use anyhow::{anyhow, ensure, Context, Result};
use axum::extract::State;
use axum::Json;
use chrono::{DateTime, Duration, NaiveTime, Utc};
use serde_json::{json, Map, Value};
use tempfile::TempDir;
use tokio::runtime::Runtime;
use uuid::Uuid;

use veld::config::{ServerConfig, StorageBackend};
use veld::handlers::recall::{proactive_context, ProactiveContextRequest};
use veld::handlers::MultiUserMemoryManager;
use veld::keystore::{store_keystore_path, KdfParams, Keystore};
use veld::memory::{
    ContextId, EmotionalContext, EpisodeContext, Experience, ExperienceType, ForgetCriteria,
    LayerMode, Memory, MemoryId, NerEntityRecord, Query, RetrievalMode, RichContext, SourceContext,
    BM25_INDEX_DIR,
};
use veld::recall_harness::fixtures::{self, CorpusItem, SmokeCase};
use veld::recall_harness::report::SMOKE_K;
use veld::recall_harness::runner::{ingest_corpus, EVAL_USER, HARNESS_CLOCK_ANCHOR};
use veld::user_memory::SharedUserMemory;

const STRUCTURED_USER: &str = "recall-parity";
const SCORE_TOLERANCE: f64 = 1e-6;
const RECALL_DEPTH: usize = 10;
const PROACTIVE_DEPTH: usize = 8;
const EXHAUSTIVE_LIMIT: usize = 10_000;
const STRUCTURED_SESSIONS: usize = 24;
const SEALED_SESSIONS: usize = 8;
const TURN_SPACING_SECONDS: i64 = 97;
const BYID_STRIDE: usize = 37;
const LOW_IMPORTANCE_QUANTILE: f32 = 0.05;
const DIFFERENCES_SHOWN: usize = 8;
const SETTLE_LIMIT: StdDuration = StdDuration::from_secs(120);
const SETTLE_POLL: StdDuration = StdDuration::from_millis(5);
const CONTEXT_NAMESPACE: u128 = 0x5eed_0007_0000_4000_8000_0000_0000_0000;
const KEYSTORE_PASSPHRASE: &str = "recall-parity-correct-horse-battery-staple-R7";

/// Each topical tag a structured memory carries when its content contains one of the cues.
const TOPICS: [(&str, &[&str]); 7] = [
    ("gaming", &["game", "tournament", "console"]),
    ("film", &["movie", "film", "screenplay", "script"]),
    ("pets", &["turtle", "dog", "pet"]),
    ("reading", &["book", "novel", "read"]),
    (
        "cooking",
        &["ice cream", "recipe", "cook", "bake", "dessert", "cake"],
    ),
    ("outdoors", &["hike", "hiking", "trail", "falls"]),
    ("writing", &["writ", "story"]),
];

/// Fields of a surfaced `proactive_context` memory that the comparison keeps (normalisation 4).
const PROACTIVE_MEMORY_FIELDS: [&str; 11] = [
    "memory_type",
    "importance",
    "created_at",
    "tags",
    "tier",
    "relevance_reason",
    "matched_entities",
    "access_count",
    "activation_level",
    "retrieval_trigger",
    "competition_conflict",
];

/// Fields of a semantic fact that name the fact or stamp the wall clock (normalisation 5).
const FACT_IDENTITY_FIELDS: [&str; 8] = [
    "id",
    "created_at",
    "last_reinforced",
    "valid_from",
    "valid_until",
    "superseded_by",
    "supersedes",
    "purged_at",
];

// ─── process environment ────────────────────────────────────────────────────────

/// Pinned by every test. Removed again on exit.
const PINNED_VARIABLES: [&str; 11] = [
    "VELD_EMBEDDER",
    "VELD_MODEL_PATH",
    "VELD_ALLOW_SIMPLIFIED_EMBEDDINGS",
    "VELD_OFFLINE",
    "VELD_EMBEDDING_API_URL",
    "VELD_ONNX_THREADS",
    "RAYON_NUM_THREADS",
    "VELD_RECALL_READONLY",
    "VELD_LAZY_CROSS_ENCODER",
    "SHODH_ALLOW_FALLBACK_NER",
    "VELD_EVAL_NOW",
];

/// Removed by every test: key material, and the levers that would move the pipeline off the
/// path `recall-eval` measures.
const CLEARED_VARIABLES: [&str; 10] = [
    "VELD_MASTER_PASSPHRASE",
    "VELD_KEYSTORE_PATH",
    "VELD_KMS_WRAP_KEY",
    "VELD_ENCRYPTION_KEY",
    "SHODH_EVAL_NOW",
    "SHODH_RECALL_READONLY",
    "SHODH_EVAL_CONSOLIDATE",
    "SHODH_EVAL_NAME_EMB",
    "VELD_NEURAL_NER",
    "VELD_QUERY_NER",
];

static PROCESS_ENVIRONMENT: Mutex<()> = Mutex::new(());

/// Whether a corpus is written in plaintext or sealed under encryption-v2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CorpusKind {
    Plaintext,
    Sealed,
}

impl CorpusKind {
    fn is_sealed(self) -> bool {
        self == CorpusKind::Sealed
    }
}

/// Exclusive use of the process environment for one test, pinned on entry and cleared on drop.
struct PinnedEnvironment {
    _exclusive: MutexGuard<'static, ()>,
    _empty_model_dir: TempDir,
}

impl PinnedEnvironment {
    fn enter(scoring_clock: DateTime<Utc>, kind: CorpusKind) -> Result<Self> {
        let exclusive = PROCESS_ENVIRONMENT
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        clear_test_variables();
        let empty_model_dir = TempDir::new()?;
        let pins: [(&str, String); 11] = [
            ("VELD_EMBEDDER", "minilm".to_string()),
            (
                "VELD_MODEL_PATH",
                empty_model_dir.path().display().to_string(),
            ),
            ("VELD_ALLOW_SIMPLIFIED_EMBEDDINGS", "1".to_string()),
            ("VELD_OFFLINE", "true".to_string()),
            ("VELD_EMBEDDING_API_URL", "http://127.0.0.1:9".to_string()),
            ("VELD_ONNX_THREADS", "1".to_string()),
            ("RAYON_NUM_THREADS", "1".to_string()),
            ("VELD_RECALL_READONLY", "1".to_string()),
            ("VELD_LAZY_CROSS_ENCODER", "1".to_string()),
            ("SHODH_ALLOW_FALLBACK_NER", "1".to_string()),
            ("VELD_EVAL_NOW", scoring_clock.to_rfc3339()),
        ];
        for (name, value) in pins {
            env::set_var(name, value);
        }
        if kind.is_sealed() {
            env::set_var("VELD_MASTER_PASSPHRASE", KEYSTORE_PASSPHRASE);
        }
        Ok(Self {
            _exclusive: exclusive,
            _empty_model_dir: empty_model_dir,
        })
    }
}

impl Drop for PinnedEnvironment {
    fn drop(&mut self) {
        clear_test_variables();
    }
}

fn clear_test_variables() {
    for name in PINNED_VARIABLES.iter().chain(CLEARED_VARIABLES.iter()) {
        env::remove_var(name);
    }
}

/// Midnight UTC of the current day: never later than the wall clock, so graph state stamped
/// with the wall clock during ingest has a non-negative age and is read undecayed, and fixed in
/// time of day, so the corpus falls on the same calendar boundaries on every run.
fn start_of_today() -> DateTime<Utc> {
    Utc::now().date_naive().and_time(NaiveTime::MIN).and_utc()
}

fn gate_scoring_clock() -> Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(HARNESS_CLOCK_ANCHOR)
        .context("HARNESS_CLOCK_ANCHOR is RFC 3339")?
        .with_timezone(&Utc))
}

/// The lowest Argon2id cost the keystore accepts. Every sealed store open pays one derivation
/// under the keystore's own parameters; the format, the unseal and the record envelope are the
/// production ones.
fn install_keystore(path: &Path) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let kdf = KdfParams {
        m_cost: 8 * 1024,
        t_cost: 1,
        ..KdfParams::production()
    };
    Keystore::create(KEYSTORE_PASSPHRASE, kdf)?.save_to_path(path)
}

fn background_runtime() -> Result<Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("building the runtime the HTTP handlers run on")
}

/// `proactive_context` strengthens the synapses between the memories it surfaced in a detached
/// task. Waiting for it keeps the next probe from racing it.
fn settle_background_tasks(runtime: &Runtime) -> Result<()> {
    let deadline = Instant::now() + SETTLE_LIMIT;
    while runtime.metrics().num_alive_tasks() > 0 {
        ensure!(
            Instant::now() < deadline,
            "{} background task(s) still running {SETTLE_LIMIT:?} after a handler returned",
            runtime.metrics().num_alive_tasks()
        );
        std::thread::sleep(SETTLE_POLL);
    }
    Ok(())
}

// ─── tenants ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Arm {
    label: &'static str,
    backend: StorageBackend,
}

const REFERENCE: Arm = Arm {
    label: "rocksdb",
    backend: StorageBackend::RocksDb,
};
const CONTROL: Arm = Arm {
    label: "rocksdb-control",
    backend: StorageBackend::RocksDb,
};
const CANDIDATE: Arm = Arm {
    label: "redb",
    backend: StorageBackend::Redb,
};

/// Corpus key ↔ memory id for one tenant (normalisation 1).
#[derive(Debug, Default)]
struct CorpusIds {
    by_key: HashMap<String, MemoryId>,
    by_id: HashMap<MemoryId, String>,
}

impl CorpusIds {
    fn insert(&mut self, key: String, id: MemoryId) {
        self.by_id.entry(id.clone()).or_insert_with(|| key.clone());
        self.by_key.insert(key, id);
    }

    fn key_of(&self, id: &MemoryId) -> String {
        self.by_id
            .get(id)
            .cloned()
            .unwrap_or_else(|| format!("unmapped:{}", id.0))
    }

    fn id_of(&self, key: &str) -> Result<MemoryId> {
        self.by_key
            .get(key)
            .cloned()
            .with_context(|| format!("corpus key {key} was never remembered"))
    }

    fn sorted_keys<'a>(&self, ids: impl IntoIterator<Item = &'a MemoryId>) -> Vec<String> {
        let mut keys: Vec<String> = ids.into_iter().map(|id| self.key_of(id)).collect();
        keys.sort();
        keys
    }
}

/// One tenant on one engine, laid out and wired as the server wires it.
struct Tenant {
    arm: Arm,
    user: &'static str,
    tenant_dir: PathBuf,
    manager: Arc<MultiUserMemoryManager>,
    memory: SharedUserMemory,
    ids: CorpusIds,
}

impl Tenant {
    fn open(arm: Arm, root: &Path, user: &'static str, kind: CorpusKind) -> Result<Self> {
        let base = root.join(arm.label);
        fs::create_dir_all(&base)?;
        let config = ServerConfig {
            storage_path: base.clone(),
            backup_enabled: false,
            effective_storage_backend: arm.backend,
            ..ServerConfig::default()
        };
        let manager = Arc::new(
            MultiUserMemoryManager::new(base.clone(), config)
                .with_context(|| format!("starting the {} manager", arm.label))?,
        );
        let tenant_dir = base.join(user);
        if kind.is_sealed() {
            install_keystore(&store_keystore_path(&tenant_dir, arm.backend))?;
        }
        let memory = manager
            .get_user_memory(user)
            .with_context(|| format!("opening the {} tenant", arm.label))?;
        Ok(Self {
            arm,
            user,
            tenant_dir,
            manager,
            memory,
            ids: CorpusIds::default(),
        })
    }

    /// The gate's own ingest path: NER, merged entity names, `remember`, then the entity graph.
    fn ingest_gate_corpus(&mut self, corpus: &[CorpusItem]) -> Result<()> {
        let id_map = ingest_corpus(&self.manager, corpus)
            .with_context(|| format!("ingesting the gate corpus into {}", self.arm.label))?;
        ensure!(
            id_map.len() == corpus.len(),
            "{} ingested {} of {} gate items",
            self.arm.label,
            id_map.len(),
            corpus.len()
        );
        for (key, uuid) in id_map {
            self.ids.insert(key, MemoryId(uuid));
        }
        Ok(())
    }

    /// One structured memory through the path the gate ingest takes, with its episode context
    /// and parent link.
    fn remember_structured(&mut self, item: &StructuredItem) -> Result<()> {
        let experience = item.experience(ner_records(&self.manager, &item.content));
        let outcome = self
            .memory
            .read()
            .remember_detailed(experience.clone(), Some(item.created_at))
            .with_context(|| format!("{} remembering {}", self.arm.label, item.key))?;
        ensure!(
            !outcome.deduped,
            "{}: {} merged into an existing memory",
            self.arm.label,
            item.key
        );
        self.manager
            .process_experience_into_graph(self.user, &experience, &outcome.id)
            .with_context(|| format!("{} graph-processing {}", self.arm.label, item.key))?;
        if let Some(parent) = &item.parent {
            let parent_id = self.ids.id_of(parent)?;
            self.memory
                .read()
                .set_memory_parent(&outcome.id, Some(parent_id))
                .with_context(|| format!("{} parenting {}", self.arm.label, item.key))?;
        }
        self.ids.insert(item.key.clone(), outcome.id);
        Ok(())
    }

    /// A lost BM25 commit means the index misses documents and every comparison after it would
    /// measure the loss, not the engine.
    fn ensure_bm25_intact(&self) -> Result<()> {
        let lost = self.memory.read().bm25_commit_failure_count();
        ensure!(
            lost == 0,
            "{} lost {lost} BM25 commit batch(es) during ingest; move the temporary directory \
             off any path a search indexer or sync daemon watches",
            self.arm.label
        );
        Ok(())
    }

    /// The tenant runs sealed: its store keystore is in place and its BM25 index stays in memory.
    fn ensure_sealed(&self) -> Result<()> {
        let keystore = store_keystore_path(&self.tenant_dir, self.arm.backend);
        ensure!(
            keystore.exists(),
            "{}: no store keystore at {}",
            self.arm.label,
            keystore.display()
        );
        let bm25 = self.tenant_dir.join(BM25_INDEX_DIR);
        ensure!(
            !bm25.exists(),
            "{}: the sealed tenant wrote its BM25 index to {}",
            self.arm.label,
            bm25.display()
        );
        Ok(())
    }

    fn observe_recall(&self, query: &Query, annotate_ner: bool) -> Result<Value> {
        let mut query = query.clone();
        if annotate_ner {
            self.manager.annotate_query_ner(&mut query);
        }
        let memory = self.memory.read();
        let traced_before: HashSet<String> = memory
            .retrieval_trace_store()
            .load_pending()
            .into_iter()
            .map(|trace| trace.trace_id)
            .collect();
        let results = memory.recall(&query)?;
        let attributions = memory.last_signal_attributions();
        let traces: Vec<_> = memory
            .retrieval_trace_store()
            .load_pending()
            .into_iter()
            .filter(|trace| !traced_before.contains(&trace.trace_id))
            .collect();

        let hits: Vec<RankedHit> = results
            .iter()
            .map(|hit| RankedHit::of_memory(hit, &self.ids))
            .collect();
        let returned: HashSet<&str> = hits.iter().map(|hit| hit.key.as_str()).collect();

        let mut signal_attribution = Map::new();
        for (id, attribution) in attributions {
            let key = self.ids.key_of(&id);
            if returned.contains(key.as_str()) {
                signal_attribution.insert(key, serde_json::to_value(&attribution)?);
            }
        }
        let mut scoring_features = Map::new();
        for trace in traces {
            scoring_features.insert(
                self.ids.key_of(&trace.memory_id),
                json!({
                    "features": serde_json::to_value(&trace.features)?,
                    "base_score": trace.base_score,
                    "final_score": trace.final_score,
                }),
            );
        }
        Ok(json!({
            "ranking": ranking_value(canonical_ranking(hits)),
            "signal_attribution": signal_attribution,
            "scoring_features": scoring_features,
        }))
    }

    fn observe_storage_order(&self, memories: &[Memory]) -> Value {
        json!({
            "count": memories.len(),
            "keys": self.ids.sorted_keys(memories.iter().map(|memory| &memory.id)),
        })
    }

    fn observe_proactive(&self, runtime: &Runtime, context: &str) -> Result<Value> {
        let request: ProactiveContextRequest = serde_json::from_value(json!({
            "user_id": self.user,
            "context": context,
            "max_results": PROACTIVE_DEPTH,
            "auto_ingest": false,
        }))?;
        let response = runtime
            .block_on(proactive_context(
                State(Arc::clone(&self.manager)),
                Json(request),
            ))
            .map_err(|e| anyhow!("proactive_context on {}: {e:?}", self.arm.label))?;
        settle_background_tasks(runtime)?;
        normalise_proactive(&serde_json::to_value(&response.0)?, &self.ids)
    }

    fn observe_inventory(&self) -> Result<Value> {
        let memories = self.memory.read().get_all_memories()?;
        let mut inventory = Map::new();
        for memory in &memories {
            inventory.insert(
                self.ids.key_of(&memory.id),
                json!({
                    "tier": format!("{:?}", memory.tier),
                    "importance": memory.importance(),
                    "access_count": memory.access_count(),
                    "forgotten": memory.is_forgotten(),
                    "parent": memory.parent_id.as_ref().map(|parent| self.ids.key_of(parent)),
                }),
            );
        }
        Ok(json!({ "count": memories.len(), "memories": inventory }))
    }

    fn observe_statistics(&self) -> Result<Value> {
        Ok(json!({
            "memory": serde_json::to_value(self.memory.read().stats())?,
            "graph": serde_json::to_value(self.manager.get_user_graph_stats(self.user)?)?,
        }))
    }

    fn observe_facts(&self) -> Result<Value> {
        let facts = self.memory.read().get_facts(self.user, EXHAUSTIVE_LIMIT)?;
        let mut normalised = Vec::with_capacity(facts.len());
        for fact in &facts {
            let mut value = serde_json::to_value(fact)?;
            strip_fields(&mut value, &FACT_IDENTITY_FIELDS);
            value["source_memories"] = json!(self.ids.sorted_keys(&fact.source_memories));
            normalised.push(value);
        }
        normalised.sort_by_cached_key(Value::to_string);
        Ok(json!({ "count": facts.len(), "facts": normalised }))
    }

    fn apply(&self, mutation: &ResolvedMutation) -> Result<Value> {
        match mutation {
            ResolvedMutation::Maintenance => Ok(json!({
                "processed": self.manager.run_maintenance_all_users(),
            })),
            ResolvedMutation::Forget(forget) => {
                let memory = self.memory.read();
                let mut forgotten = 0usize;
                for criteria in forget.criteria(&self.ids)? {
                    forgotten += memory.forget(criteria)?;
                }
                Ok(json!({ "forgotten": forgotten }))
            }
        }
    }

    fn importance_quantile(&self, quantile: f32) -> Result<f32> {
        let mut importances: Vec<f32> = self
            .memory
            .read()
            .get_all_memories()?
            .iter()
            .filter(|memory| !memory.is_forgotten())
            .map(|memory| memory.importance())
            .collect();
        ensure!(!importances.is_empty(), "no memories to take a quantile of");
        importances.sort_by(f32::total_cmp);
        let index = ((importances.len() - 1) as f32 * quantile).round() as usize;
        Ok(importances[index])
    }
}

/// The reference, the same-engine control and the candidate, always visited in that order.
struct Tenants {
    reference: Tenant,
    control: Tenant,
    candidate: Tenant,
}

impl Tenants {
    fn open(root: &Path, user: &'static str, kind: CorpusKind) -> Result<Self> {
        Ok(Self {
            reference: Tenant::open(REFERENCE, root, user, kind)?,
            control: Tenant::open(CONTROL, root, user, kind)?,
            candidate: Tenant::open(CANDIDATE, root, user, kind)?,
        })
    }

    fn all(&self) -> [&Tenant; 3] {
        [&self.reference, &self.control, &self.candidate]
    }

    fn all_mut(&mut self) -> [&mut Tenant; 3] {
        [&mut self.reference, &mut self.control, &mut self.candidate]
    }

    fn observe(&self, observe: impl Fn(&Tenant) -> Result<Value>) -> Observed {
        let [reference, control, candidate] = self
            .all()
            .map(|tenant| observe(tenant).unwrap_or_else(error_value));
        Observed {
            reference,
            control,
            candidate,
        }
    }

    /// Each structured memory goes into the three tenants before the next is written, so
    /// wall-clock stamps taken during ingest are milliseconds apart across them.
    fn ingest_structured(&mut self, corpus: &StructuredCorpus) -> Result<()> {
        for item in &corpus.items {
            for tenant in self.all_mut() {
                tenant.remember_structured(item)?;
            }
        }
        self.all()
            .iter()
            .try_for_each(|tenant| tenant.ensure_bm25_intact())
    }
}

fn error_value(error: anyhow::Error) -> Value {
    json!({ "error": format!("{error:#}") })
}

fn ner_records(manager: &MultiUserMemoryManager, content: &str) -> Vec<NerEntityRecord> {
    match manager.get_neural_ner().extract(content) {
        Ok(entities) => entities
            .into_iter()
            .map(|entity| NerEntityRecord {
                text: entity.text,
                entity_type: entity.entity_type.as_str().to_string(),
                confidence: entity.confidence,
                start_char: Some(entity.start),
                end_char: Some(entity.end),
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

// ─── rankings ───────────────────────────────────────────────────────────────────

/// One ranked result: its corpus key, score and the per-memory fields compared beside it.
#[derive(Debug, Clone)]
struct RankedHit {
    key: String,
    score: Option<f32>,
    detail: Map<String, Value>,
}

impl RankedHit {
    fn of_memory(memory: &Memory, ids: &CorpusIds) -> Self {
        let mut detail = Map::new();
        detail.insert("tier".into(), json!(format!("{:?}", memory.tier)));
        detail.insert("importance".into(), json!(memory.importance()));
        Self {
            key: ids.key_of(&memory.id),
            score: memory.score,
            detail,
        }
    }
}

fn scores_tie(left: Option<f32>, right: Option<f32>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => (f64::from(left) - f64::from(right)).abs() <= SCORE_TOLERANCE,
        _ => false,
    }
}

/// Normalisation 2: each run of results tied with its neighbour is put in corpus-key order.
fn canonical_ranking(hits: Vec<RankedHit>) -> Vec<RankedHit> {
    let mut ordered = Vec::with_capacity(hits.len());
    let mut group: Vec<RankedHit> = Vec::new();
    for hit in hits {
        if group
            .last()
            .is_some_and(|last| !scores_tie(last.score, hit.score))
        {
            group.sort_by(|a, b| a.key.cmp(&b.key));
            ordered.append(&mut group);
        }
        group.push(hit);
    }
    group.sort_by(|a, b| a.key.cmp(&b.key));
    ordered.append(&mut group);
    ordered
}

fn ranking_value(hits: Vec<RankedHit>) -> Value {
    Value::Array(
        hits.into_iter()
            .map(|hit| {
                let mut entry = Map::new();
                entry.insert("key".into(), json!(hit.key));
                entry.insert("score".into(), json!(hit.score));
                entry.extend(hit.detail);
                Value::Object(entry)
            })
            .collect(),
    )
}

/// Normalisation 4 over a `ProactiveContextResponse`.
fn normalise_proactive(body: &Value, ids: &CorpusIds) -> Result<Value> {
    let surfaced = body["memories"]
        .as_array()
        .context("proactive_context returned no memories array")?;
    let mut hits = Vec::with_capacity(surfaced.len());
    for memory in surfaced {
        let id = memory["id"]
            .as_str()
            .context("a surfaced memory without an id")?;
        let id = MemoryId(Uuid::parse_str(id).with_context(|| format!("memory id {id}"))?);
        let mut detail = Map::new();
        for field in PROACTIVE_MEMORY_FIELDS {
            if let Some(value) = memory.get(field) {
                detail.insert(field.to_string(), value.clone());
            }
        }
        hits.push(RankedHit {
            key: ids.key_of(&id),
            score: memory["score"].as_f64().map(|score| score as f32),
            detail,
        });
    }
    let mut facts: Vec<Value> = body
        .get("relevant_facts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for fact in &mut facts {
        strip_fields(fact, &["id"]);
    }
    facts.sort_by_cached_key(Value::to_string);
    Ok(json!({
        "memories": ranking_value(canonical_ranking(hits)),
        "memory_count": body["memory_count"],
        "reminder_count": body["reminder_count"],
        "todo_count": body["todo_count"],
        "detected_entities": body.get("detected_entities").cloned().unwrap_or(Value::Null),
        "relevant_facts": facts,
    }))
}

fn strip_fields(value: &mut Value, fields: &[&str]) {
    if let Value::Object(map) = value {
        for field in fields {
            map.remove(*field);
        }
    }
}

// ─── comparison ─────────────────────────────────────────────────────────────────

/// One probe's output on the three tenants.
struct Observed {
    reference: Value,
    control: Value,
    candidate: Value,
}

/// Every place two observations differ, numbers within [`SCORE_TOLERANCE`].
fn differences(path: &str, left: &Value, right: &Value, out: &mut Vec<String>) {
    match (left, right) {
        (Value::Object(l), Value::Object(r)) => {
            let keys: BTreeSet<&String> = l.keys().chain(r.keys()).collect();
            for key in keys {
                let child = format!("{path}.{key}");
                match (l.get(key), r.get(key)) {
                    (Some(a), Some(b)) => differences(&child, a, b, out),
                    (Some(a), None) => out.push(format!("{child}: only on the left: {}", brief(a))),
                    (None, Some(b)) => {
                        out.push(format!("{child}: only on the right: {}", brief(b)))
                    }
                    (None, None) => {}
                }
            }
        }
        (Value::Array(l), Value::Array(r)) => {
            if l.len() != r.len() {
                out.push(format!(
                    "{path}: {} entries on the left, {} on the right",
                    l.len(),
                    r.len()
                ));
            }
            for (index, (a, b)) in l.iter().zip(r).enumerate() {
                differences(&format!("{path}[{index}]"), a, b, out);
            }
        }
        (Value::Number(l), Value::Number(r)) => match (l.as_f64(), r.as_f64()) {
            (Some(a), Some(b)) if (a - b).abs() <= SCORE_TOLERANCE => {}
            _ => out.push(format!("{path}: {l} != {r}")),
        },
        _ if left != right => out.push(format!("{path}: {} != {}", brief(left), brief(right))),
        _ => {}
    }
}

fn brief(value: &Value) -> String {
    const LIMIT: usize = 160;
    let text = value.to_string();
    if text.chars().count() <= LIMIT {
        text
    } else {
        format!("{}…", text.chars().take(LIMIT).collect::<String>())
    }
}

#[derive(Debug)]
struct Finding {
    phase: String,
    probe: String,
    differences: Vec<String>,
}

impl Finding {
    fn label(&self) -> String {
        format!("[{}] {}", self.phase, self.probe)
    }
}

/// Every comparison a scenario made, and where the tenants disagreed.
#[derive(Debug, Default)]
struct ParityLedger {
    compared: usize,
    cross_engine: Vec<Finding>,
    same_engine: Vec<Finding>,
}

impl ParityLedger {
    fn record(&mut self, phase: &str, probe: &str, observed: &Observed) {
        self.compared += 1;
        let mut cross = Vec::new();
        differences("$", &observed.reference, &observed.candidate, &mut cross);
        let mut same = Vec::new();
        differences("$", &observed.reference, &observed.control, &mut same);
        let finding = |differences| Finding {
            phase: phase.to_string(),
            probe: probe.to_string(),
            differences,
        };
        if !cross.is_empty() {
            self.cross_engine.push(finding(cross));
        }
        if !same.is_empty() {
            self.same_engine.push(finding(same));
        }
    }

    fn report(&self, scenario: &str) -> String {
        let nondeterministic: HashSet<String> =
            self.same_engine.iter().map(Finding::label).collect();
        let mut report = format!(
            "{scenario}: {} comparisons; {} differ between {} and {} (engine), {} between {} and \
             {} (same-engine control)",
            self.compared,
            self.cross_engine.len(),
            REFERENCE.label,
            CANDIDATE.label,
            self.same_engine.len(),
            REFERENCE.label,
            CONTROL.label,
        );
        let sections = [
            (
                format!("{} vs {}", REFERENCE.label, CANDIDATE.label),
                &self.cross_engine,
            ),
            (
                format!("{} vs {}", REFERENCE.label, CONTROL.label),
                &self.same_engine,
            ),
        ];
        for (title, findings) in sections {
            if findings.is_empty() {
                continue;
            }
            let _ = write!(report, "\n{title}:");
            for finding in findings {
                let marker = if title.ends_with(CANDIDATE.label)
                    && nondeterministic.contains(&finding.label())
                {
                    " (also differs on the same engine: nondeterministic)"
                } else {
                    ""
                };
                let _ = write!(
                    report,
                    "\n  {}{marker}: {} difference(s)",
                    finding.label(),
                    finding.differences.len()
                );
                for line in finding.differences.iter().take(DIFFERENCES_SHOWN) {
                    let _ = write!(report, "\n      {line}");
                }
            }
        }
        report
    }

    fn assert_clean(&self, scenario: &str) {
        let report = self.report(scenario);
        eprintln!("{report}");
        assert!(
            self.cross_engine.is_empty() && self.same_engine.is_empty(),
            "{report}"
        );
    }
}

// ─── probes and mutations ───────────────────────────────────────────────────────

/// A read run on every tenant and compared.
enum Probe {
    Recall {
        name: String,
        query: Box<Query>,
        annotate_ner: bool,
    },
    TagRecall {
        name: String,
        tags: Vec<String>,
    },
    DateRecall {
        name: String,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },
    Proactive {
        name: String,
        context: String,
    },
    Inventory,
    Statistics,
    Facts,
}

impl Probe {
    fn recall(name: &str, query: Query) -> Self {
        Probe::Recall {
            name: format!("recall:{name}"),
            query: Box::new(query),
            annotate_ner: true,
        }
    }

    fn filter(name: &str, query: Query) -> Self {
        Probe::Recall {
            name: format!("filter:{name}"),
            query: Box::new(query),
            annotate_ner: false,
        }
    }

    fn name(&self) -> String {
        match self {
            Probe::Recall { name, .. }
            | Probe::TagRecall { name, .. }
            | Probe::DateRecall { name, .. }
            | Probe::Proactive { name, .. } => name.clone(),
            Probe::Inventory => "get_all_memories".to_string(),
            Probe::Statistics => "stats".to_string(),
            Probe::Facts => "facts".to_string(),
        }
    }

    fn observe(&self, tenant: &Tenant, runtime: &Runtime) -> Result<Value> {
        match self {
            Probe::Recall {
                query,
                annotate_ner,
                ..
            } => tenant.observe_recall(query, *annotate_ner),
            Probe::TagRecall { tags, .. } => {
                let memories = tenant
                    .memory
                    .read()
                    .recall_by_tags(tags, EXHAUSTIVE_LIMIT)?;
                Ok(tenant.observe_storage_order(&memories))
            }
            Probe::DateRecall { start, end, .. } => {
                let memories =
                    tenant
                        .memory
                        .read()
                        .recall_by_date(*start, *end, EXHAUSTIVE_LIMIT)?;
                Ok(tenant.observe_storage_order(&memories))
            }
            Probe::Proactive { context, .. } => tenant.observe_proactive(runtime, context),
            Probe::Inventory => tenant.observe_inventory(),
            Probe::Statistics => tenant.observe_statistics(),
            Probe::Facts => tenant.observe_facts(),
        }
    }
}

fn run_probes(
    phase: &str,
    probes: &[Probe],
    tenants: &Tenants,
    runtime: &Runtime,
    ledger: &mut ParityLedger,
) {
    for probe in probes {
        let observed = tenants.observe(|tenant| probe.observe(tenant, runtime));
        ledger.record(phase, &probe.name(), &observed);
    }
}

/// A forgetting step, stated over corpus keys and data rather than memory ids.
#[derive(Debug, Clone)]
enum Forget {
    OlderThanDays(u32),
    LowImportance(f32),
    Keys(Vec<String>),
    Tags(Vec<String>),
    Pattern(String),
    DateRange {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },
    Type(ExperienceType),
}

impl Forget {
    fn criteria(&self, ids: &CorpusIds) -> Result<Vec<ForgetCriteria>> {
        Ok(match self {
            Forget::OlderThanDays(days) => vec![ForgetCriteria::OlderThan(*days)],
            Forget::LowImportance(threshold) => vec![ForgetCriteria::LowImportance(*threshold)],
            Forget::Keys(keys) => keys
                .iter()
                .map(|key| ids.id_of(key).map(ForgetCriteria::ById))
                .collect::<Result<_>>()?,
            Forget::Tags(tags) => vec![ForgetCriteria::ByTags(tags.clone())],
            Forget::Pattern(pattern) => vec![ForgetCriteria::Pattern(pattern.clone())],
            Forget::DateRange { start, end } => vec![ForgetCriteria::ByDateRange {
                start: *start,
                end: *end,
            }],
            Forget::Type(experience_type) => vec![ForgetCriteria::ByType(experience_type.clone())],
        })
    }
}

/// A write run on every tenant in turn, its return value compared like a probe's.
enum Mutation {
    Maintenance,
    Forget {
        name: &'static str,
        forget: Forget,
    },
    /// `LowImportance` at a quantile of the reference tenant's importances, taken when the step
    /// runs and applied to every tenant at that one threshold.
    ForgetLowImportanceQuantile {
        name: &'static str,
        quantile: f32,
    },
}

enum ResolvedMutation {
    Maintenance,
    Forget(Forget),
}

fn apply_mutation(
    phase: &str,
    mutation: &Mutation,
    tenants: &Tenants,
    ledger: &mut ParityLedger,
) -> Result<()> {
    let (name, resolved) = match mutation {
        Mutation::Maintenance => (
            "run_maintenance_all_users".to_string(),
            ResolvedMutation::Maintenance,
        ),
        Mutation::Forget { name, forget } => (
            format!("forget:{name}"),
            ResolvedMutation::Forget(forget.clone()),
        ),
        Mutation::ForgetLowImportanceQuantile { name, quantile } => {
            let threshold = tenants.reference.importance_quantile(*quantile)?;
            (
                format!("forget:{name}"),
                ResolvedMutation::Forget(Forget::LowImportance(threshold)),
            )
        }
    };
    let observed = tenants.observe(|tenant| tenant.apply(&resolved));
    ledger.record(phase, &name, &observed);
    Ok(())
}

// ─── the gate corpus ────────────────────────────────────────────────────────────

fn load_gate_corpus() -> Result<Vec<CorpusItem>> {
    Ok(fixtures::load_corpus(&fixtures::manifest_path(
        fixtures::LOCOMO_GATE_CORPUS_PATH,
    ))?)
}

fn load_gate_cases() -> Result<Vec<SmokeCase>> {
    Ok(fixtures::load_smoke_cases(&fixtures::manifest_path(
        fixtures::LOCOMO_GATE_CASES_PATH,
    ))?)
}

/// Every gate case, queried as `recall-eval` queries it, then the stored state.
fn gate_probes(cases: &[SmokeCase]) -> Vec<Probe> {
    let mut probes: Vec<Probe> = cases
        .iter()
        .map(|case| Probe::Recall {
            name: format!("gate:{}:{:?}", case.id, case.category),
            query: Box::new(Query {
                query_text: Some(case.query.clone()),
                max_results: SMOKE_K,
                layers: Some(LayerMode::Full),
                ..Default::default()
            }),
            annotate_ner: true,
        })
        .collect();
    probes.extend([Probe::Inventory, Probe::Statistics]);
    probes
}

// ─── the structured corpus ──────────────────────────────────────────────────────

/// One gate turn with episode, timing, tags, type, affect, credibility and parent assigned.
#[derive(Debug, Clone)]
struct StructuredItem {
    key: String,
    ordinal: u32,
    episode: String,
    turn: u32,
    speaker: String,
    content: String,
    tags: Vec<String>,
    created_at: DateTime<Utc>,
    episode_start: DateTime<Utc>,
    experience_type: ExperienceType,
    parent: Option<String>,
}

impl StructuredItem {
    fn experience(&self, ner_entities: Vec<NerEntityRecord>) -> Experience {
        let mut entities = vec![self.speaker.clone()];
        let mut seen: HashSet<String> = HashSet::from([self.speaker.to_lowercase()]);
        for record in &ner_entities {
            if seen.insert(record.text.to_lowercase()) {
                entities.push(record.text.clone());
            }
        }
        Experience {
            experience_type: self.experience_type.clone(),
            content: self.content.clone(),
            context: Some(self.context()),
            entities,
            tags: self.tags.clone(),
            metadata: HashMap::from([("speaker".to_string(), self.speaker.clone())]),
            ner_entities,
            ..Default::default()
        }
    }

    fn context(&self) -> RichContext {
        let ordinal = self.ordinal;
        RichContext {
            id: ContextId(Uuid::from_u128(CONTEXT_NAMESPACE + u128::from(ordinal))),
            conversation: Default::default(),
            user: Default::default(),
            project: Default::default(),
            temporal: Default::default(),
            semantic: Default::default(),
            code: Default::default(),
            document: Default::default(),
            environment: Default::default(),
            emotional: EmotionalContext {
                valence: ((ordinal * 53) % 201) as f32 / 100.0 - 1.0,
                arousal: ((ordinal * 37) % 101) as f32 / 100.0,
                ..Default::default()
            },
            source: SourceContext {
                credibility: 0.5 + ((ordinal * 29) % 50) as f32 / 100.0,
                ..Default::default()
            },
            episode: EpisodeContext {
                episode_id: Some(self.episode.clone()),
                sequence_number: Some(self.turn),
                episode_type: Some("conversation".to_string()),
                episode_start: Some(self.episode_start),
                is_episode_start: self.turn == 1,
                ..Default::default()
            },
            place: Default::default(),
            who: Default::default(),
            why: Default::default(),
            binding: Default::default(),
            session: Default::default(),
            parent: None,
            embeddings: None,
            decay_rate: 0.1,
            created_at: self.created_at,
            updated_at: self.created_at,
        }
    }
}

#[derive(Debug, Clone)]
struct Session {
    episode: String,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
}

/// The first `session_count` sessions of the LoCoMo gate conversation, re-timed so the last
/// session starts a day before the scoring clock and each session keeps its original spacing
/// (January to October), with turns 97 s apart inside it. Topical tags come from content cues,
/// the experience type from the turn number, and every fourth turn is the child of the turn
/// before it.
struct StructuredCorpus {
    items: Vec<StructuredItem>,
    sessions: Vec<Session>,
}

impl StructuredCorpus {
    fn build(
        gate: &[CorpusItem],
        session_count: usize,
        scoring_clock: DateTime<Utc>,
    ) -> Result<Self> {
        let mut episodes: Vec<(String, Vec<(u32, &CorpusItem)>)> = Vec::new();
        for item in gate {
            let (episode, turn) = item
                .id
                .rsplit_once(':')
                .with_context(|| format!("gate id {} has no turn", item.id))?;
            let turn: u32 = turn
                .parse()
                .with_context(|| format!("gate id {}", item.id))?;
            match episodes.last_mut() {
                Some(current) if current.0 == episode => current.1.push((turn, item)),
                _ => {
                    if episodes.len() == session_count {
                        break;
                    }
                    episodes.push((episode.to_string(), vec![(turn, item)]));
                }
            }
        }
        ensure!(
            episodes.len() == session_count,
            "the gate corpus has {} sessions, {session_count} were asked for",
            episodes.len()
        );

        let last_start = episodes[session_count - 1].1[0].1.created_at;
        let shift = scoring_clock - Duration::days(1) - last_start;
        let mut items = Vec::new();
        let mut sessions = Vec::with_capacity(session_count);
        for (episode, turns) in &episodes {
            let start = turns[0].1.created_at + shift;
            let mut end = start;
            for (turn, item) in turns {
                let created_at =
                    start + Duration::seconds(TURN_SPACING_SECONDS * (i64::from(*turn) - 1));
                end = end.max(created_at);
                let speaker = item
                    .content
                    .split_once(": ")
                    .map(|(speaker, _)| speaker.to_string())
                    .with_context(|| format!("gate turn {} names no speaker", item.id))?;
                let mut tags = item.tags.clone();
                let lowered = item.content.to_lowercase();
                for (tag, cues) in TOPICS {
                    if cues.iter().any(|cue| lowered.contains(cue)) {
                        tags.push(tag.to_string());
                    }
                }
                items.push(StructuredItem {
                    key: item.id.clone(),
                    ordinal: items.len() as u32,
                    episode: episode.clone(),
                    turn: *turn,
                    speaker,
                    content: item.content.clone(),
                    tags,
                    created_at,
                    episode_start: start,
                    experience_type: structured_type(*turn),
                    parent: (turn % 4 == 0).then(|| format!("{episode}:{}", turn - 1)),
                });
            }
            sessions.push(Session {
                episode: episode.clone(),
                start,
                end,
            });
        }
        let keys: HashSet<&str> = items.iter().map(|item| item.key.as_str()).collect();
        for item in &items {
            if let Some(parent) = &item.parent {
                ensure!(
                    keys.contains(parent.as_str()),
                    "{} names parent {parent}, which is not in the corpus",
                    item.key
                );
            }
        }
        Ok(Self { items, sessions })
    }

    /// Session `number`, counted from 1.
    fn session(&self, number: usize) -> &Session {
        &self.sessions[number - 1]
    }

    fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

fn structured_type(turn: u32) -> ExperienceType {
    match turn % 9 {
        0 => ExperienceType::Decision,
        3 => ExperienceType::Learning,
        5 => ExperienceType::Discovery,
        7 => ExperienceType::Observation,
        8 => ExperienceType::Task,
        _ => ExperienceType::Conversation,
    }
}

fn midnight(at: DateTime<Utc>) -> DateTime<Utc> {
    at.date_naive().and_time(NaiveTime::MIN).and_utc()
}

fn semantic_query(text: &str) -> Query {
    Query {
        user_id: Some(STRUCTURED_USER.to_string()),
        query_text: Some(text.to_string()),
        max_results: RECALL_DEPTH,
        ..Default::default()
    }
}

fn filter_query() -> Query {
    Query {
        user_id: Some(STRUCTURED_USER.to_string()),
        max_results: EXHAUSTIVE_LIMIT,
        ..Default::default()
    }
}

/// 35 recalls (factual, exploratory, time-scoped, ordinal-session, episode-scoped, tag-scoped,
/// date-range, filtered, spreading activation and filter-only), four `recall_by_tags`, three
/// `recall_by_date`, five `proactive_context`, and the stored state.
fn structured_probes(corpus: &StructuredCorpus, scoring_clock: DateTime<Utc>) -> Vec<Probe> {
    let count = corpus.session_count();
    let quarter = (count / 4).max(1);
    let half = (count / 2).max(1);
    let three_quarters = (count * 3 / 4).max(1);
    let tags = |names: &[&str]| -> Option<Vec<String>> {
        Some(names.iter().map(|name| name.to_string()).collect())
    };

    let mut probes = Vec::new();
    for (name, text) in [
        (
            "factual:turtles-owned",
            "For how long has Nate had his turtles?",
        ),
        (
            "factual:coconut-ice-cream",
            "What did Nate think of the coconut milk ice cream he made?",
        ),
        (
            "factual:little-women",
            "What is \"Little Women\" about according to Joanna?",
        ),
        (
            "factual:screenplay-inspiration",
            "What inspired Joanna's new screenplay?",
        ),
        (
            "factual:turtles-as-pets",
            "Why does Nate like turtles as pets?",
        ),
        ("factual:audition", "What was Joanna's audition for?"),
        (
            "factual:hair-colour",
            "Why did Nate choose the hair color he did?",
        ),
        ("factual:books", "What kind of books does Nate enjoy?"),
        (
            "exploratory:shared-interests",
            "What kind of interests do Joanna and Nate share?",
        ),
        (
            "exploratory:screenplay-emotions",
            "What emotions is Joanna feeling about the screenplay she submitted?",
        ),
        ("exploratory:allergies", "What is Joanna allergic to?"),
        (
            "exploratory:other-friends",
            "Is it likely that Nate has friends besides Joanna?",
        ),
        ("exploratory:hobbies", "What are Joanna's hobbies?"),
        ("time:last-week", "What did Nate talk about last week?"),
        ("time:yesterday", "What did Joanna say yesterday?"),
        ("time:last-month", "What did they discuss last month?"),
        (
            "time:two-sessions-ago",
            "What did Nate mention two sessions ago?",
        ),
        (
            "time:three-months-ago",
            "What was going on three months ago?",
        ),
        (
            "session:second",
            "What did Joanna and Nate talk about in the second session?",
        ),
        ("session:third", "What did Nate share in the third session?"),
        ("session:fourth-meeting", "Summarize the fourth meeting"),
    ] {
        probes.push(Probe::recall(name, semantic_query(text)));
    }

    probes.push(Probe::recall(
        "episode:turtles",
        Query {
            episode_id: Some(corpus.session(half).episode.clone()),
            ..semantic_query("turtles")
        },
    ));
    probes.push(Probe::recall(
        "episode:screenplay",
        Query {
            episode_id: Some(corpus.session(2.min(count)).episode.clone()),
            ..semantic_query("screenplay")
        },
    ));
    probes.push(Probe::recall(
        "tag:cooking",
        Query {
            tags: tags(&["cooking"]),
            ..semantic_query("ice cream recipe")
        },
    ));
    probes.push(Probe::recall(
        "tag:gaming",
        Query {
            tags: tags(&["gaming"]),
            ..semantic_query("winning the tournament")
        },
    ));
    probes.push(Probe::recall(
        "tag:film-or-reading",
        Query {
            tags: tags(&["film", "reading"]),
            ..semantic_query("a movie recommendation")
        },
    ));
    probes.push(Probe::recall(
        "range:quarter-to-three-quarters",
        Query {
            time_range: Some((
                corpus.session(quarter).start,
                corpus.session(three_quarters).end,
            )),
            ..semantic_query("hiking with friends")
        },
    ));
    probes.push(Probe::recall(
        "range:last-thirty-days",
        Query {
            time_range: Some((scoring_clock - Duration::days(30), scoring_clock)),
            ..semantic_query("what Nate cooked")
        },
    ));
    probes.push(Probe::recall(
        "filtered:importance",
        Query {
            importance_threshold: Some(0.5),
            ..semantic_query("Joanna's plans")
        },
    ));
    probes.push(Probe::recall(
        "filtered:types",
        Query {
            experience_types: Some(vec![ExperienceType::Decision, ExperienceType::Learning]),
            ..semantic_query("decisions Nate made")
        },
    ));
    probes.push(Probe::recall(
        "mode:spreading-activation",
        Query {
            retrieval_mode: RetrievalMode::SpreadingActivation,
            ..semantic_query("taking the turtles for a walk")
        },
    ));
    probes.push(Probe::filter(
        "episode-sequence",
        Query {
            episode_id: Some(
                corpus
                    .session(count.saturating_sub(1).max(1))
                    .episode
                    .clone(),
            ),
            retrieval_mode: RetrievalMode::Temporal,
            ..filter_query()
        },
    ));
    probes.push(Probe::filter(
        "tags:pets",
        Query {
            tags: tags(&["pets"]),
            ..filter_query()
        },
    ));
    probes.push(Probe::filter(
        "time-range",
        Query {
            time_range: Some((
                corpus.session(half).start - Duration::hours(1),
                corpus.session((half + 2).min(count)).end + Duration::hours(1),
            )),
            retrieval_mode: RetrievalMode::Temporal,
            ..filter_query()
        },
    ));
    probes.push(Probe::filter(
        "default-week",
        Query {
            retrieval_mode: RetrievalMode::Temporal,
            ..filter_query()
        },
    ));

    for (name, names) in [
        ("gaming", &["gaming"][..]),
        ("pets-or-cooking", &["pets", "cooking"][..]),
        ("speaker-joanna", &["Joanna"][..]),
        ("film-reading-writing", &["film", "reading", "writing"][..]),
    ] {
        probes.push(Probe::TagRecall {
            name: format!("recall_by_tags:{name}"),
            tags: names.iter().map(|name| name.to_string()).collect(),
        });
    }

    let unaligned_from = corpus.session(count.saturating_sub(4).max(1));
    let unaligned_to = corpus.session(count.saturating_sub(2).max(1));
    probes.extend([
        Probe::DateRecall {
            name: "recall_by_date:calendar-days".to_string(),
            start: midnight(corpus.session(quarter).start),
            end: midnight(corpus.session(half).start) + Duration::days(1) - Duration::seconds(1),
        },
        Probe::DateRecall {
            name: "recall_by_date:mid-session-bounds".to_string(),
            start: unaligned_from.start + Duration::minutes(5),
            end: unaligned_to.start + Duration::minutes(5),
        },
        Probe::DateRecall {
            name: "recall_by_date:everything".to_string(),
            start: scoring_clock - Duration::days(400),
            end: scoring_clock,
        },
    ]);

    for (name, context) in [
        (
            "turtle-care",
            "Nate is thinking about getting another turtle and wonders how to care for it",
        ),
        (
            "screenplay-nerves",
            "Joanna is working on a new screenplay and feels nervous about submitting it",
        ),
        (
            "gaming-party",
            "Planning a gaming party with friends and some vegan ice cream",
        ),
        (
            "hiking",
            "Going hiking on some new trails near the falls this weekend",
        ),
        (
            "recommendation",
            "Looking for a good book or movie recommendation for tonight",
        ),
    ] {
        probes.push(Probe::Proactive {
            name: format!("proactive_context:{name}"),
            context: context.to_string(),
        });
    }

    probes.extend([Probe::Inventory, Probe::Statistics, Probe::Facts]);
    probes
}

/// Soft forgets by age, hard deletes by tag, pattern, id, calendar day and type, then a soft
/// forget by importance. The age cut falls in the gap between sessions 1 and 2, so session 1 is
/// soft-forgotten before the tag and pattern deletes run over it.
fn forgetting_plan(corpus: &StructuredCorpus) -> Result<Vec<Mutation>> {
    let first = corpus.session(1);
    let second = corpus.session(2);
    let now = Utc::now();
    let days = (now - second.start).num_days() + 1;
    let cutoff = now - Duration::days(days);
    ensure!(
        first.end < cutoff && cutoff < second.start,
        "the age cut {cutoff} does not fall between session 1 (ends {}) and session 2 (starts {})",
        first.end,
        second.start
    );
    let days = u32::try_from(days).context("an age cut in days")?;

    let by_id: Vec<String> = corpus
        .items
        .iter()
        .filter(|item| item.episode != first.episode)
        .filter(|item| !item.tags.iter().any(|tag| tag == "gaming"))
        .filter(|item| !item.content.to_lowercase().contains("turtle"))
        .step_by(BYID_STRIDE)
        .map(|item| item.key.clone())
        .collect();
    let day = midnight(corpus.session((corpus.session_count() / 2).max(3)).start);

    Ok(vec![
        Mutation::Forget {
            name: "older-than-session-2",
            forget: Forget::OlderThanDays(days),
        },
        Mutation::Forget {
            name: "tags:gaming",
            forget: Forget::Tags(vec!["gaming".to_string()]),
        },
        Mutation::Forget {
            name: "pattern:turtles",
            forget: Forget::Pattern("(?i)\\bturtles?\\b".to_string()),
        },
        Mutation::Forget {
            name: "by-id",
            forget: Forget::Keys(by_id),
        },
        Mutation::Forget {
            name: "calendar-day",
            forget: Forget::DateRange {
                start: day,
                end: day + Duration::days(1) - Duration::seconds(1),
            },
        },
        Mutation::Forget {
            name: "type:discovery",
            forget: Forget::Type(ExperienceType::Discovery),
        },
        Mutation::ForgetLowImportanceQuantile {
            name: "low-importance",
            quantile: LOW_IMPORTANCE_QUANTILE,
        },
    ])
}

/// Ingest, then probe; a heavy maintenance cycle, then probe; the forgetting plan, then probe.
fn run_structured_scenario(kind: CorpusKind, session_count: usize, scenario: &str) -> Result<()> {
    let scoring_clock = start_of_today();
    let _environment = PinnedEnvironment::enter(scoring_clock, kind)?;
    let corpus = StructuredCorpus::build(&load_gate_corpus()?, session_count, scoring_clock)?;
    let root = TempDir::new()?;
    let mut tenants = Tenants::open(root.path(), STRUCTURED_USER, kind)?;
    tenants.ingest_structured(&corpus)?;
    if kind.is_sealed() {
        tenants
            .all()
            .iter()
            .try_for_each(|tenant| tenant.ensure_sealed())?;
    }

    let runtime = background_runtime()?;
    let probes = structured_probes(&corpus, scoring_clock);
    let mut ledger = ParityLedger::default();

    run_probes("after ingest", &probes, &tenants, &runtime, &mut ledger);

    apply_mutation("maintenance", &Mutation::Maintenance, &tenants, &mut ledger)?;
    run_probes(
        "after maintenance",
        &probes,
        &tenants,
        &runtime,
        &mut ledger,
    );

    for mutation in forgetting_plan(&corpus)? {
        apply_mutation("forgetting", &mutation, &tenants, &mut ledger)?;
    }
    run_probes("after forgetting", &probes, &tenants, &runtime, &mut ledger);

    ledger.assert_clean(scenario);
    Ok(())
}

// ─── tests ──────────────────────────────────────────────────────────────────────

/// The LoCoMo gate corpus (629 turns) and its 100 cases, ingested through the gate's own ingest
/// and queried at `max_results = SMOKE_K` on the full pipeline under the gate's scoring clock.
#[test]
fn locomo_gate_cases_rank_and_score_identically_on_both_engines() -> Result<()> {
    let _environment = PinnedEnvironment::enter(gate_scoring_clock()?, CorpusKind::Plaintext)?;
    let corpus = load_gate_corpus()?;
    let cases = load_gate_cases()?;
    let root = TempDir::new()?;
    let mut tenants = Tenants::open(root.path(), EVAL_USER, CorpusKind::Plaintext)?;
    for tenant in tenants.all_mut() {
        tenant.ingest_gate_corpus(&corpus)?;
        tenant.ensure_bm25_intact()?;
    }

    let runtime = background_runtime()?;
    let mut ledger = ParityLedger::default();
    run_probes(
        "after ingest",
        &gate_probes(&cases),
        &tenants,
        &runtime,
        &mut ledger,
    );
    ledger.assert_clean("LoCoMo gate");
    Ok(())
}

/// 24 gate sessions (490 memories) with episodes, sequence numbers, topical tags, entities,
/// parents, affect, credibility and seven experience types, spanning nine months: every probe
/// after ingest, after a heavy maintenance cycle (tier promotion, activation decay, fact
/// extraction, replay, graph decay), and after soft and hard forgetting.
#[test]
fn structured_recall_is_identical_on_both_engines_through_maintenance_and_forgetting() -> Result<()>
{
    run_structured_scenario(
        CorpusKind::Plaintext,
        STRUCTURED_SESSIONS,
        "structured plaintext corpus",
    )
}

/// The same scenario over eight sessions (164 memories) sealed under encryption-v2: a store
/// keystore unsealed by `VELD_MASTER_PASSPHRASE`, sealed records, and the BM25 index held in
/// memory.
#[test]
fn sealed_structured_recall_is_identical_on_both_engines() -> Result<()> {
    run_structured_scenario(
        CorpusKind::Sealed,
        SEALED_SESSIONS,
        "structured encryption-v2 corpus",
    )
}
