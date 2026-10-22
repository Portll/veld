//! Differential harness (ADR 0007): one generated operation sequence applied to every
//! engine, and every observable compared after every step.
//!
//! With both engines built, the subjects are one store per engine, and every difference
//! is a defect. [`KNOWN_DIVERGENCES`] names what the engines still define differently
//! and the generator therefore never produces; nothing is admitted at comparison time.
//! With one engine built, the second subject is a fresh instance of the same engine
//! replaying the sequence from scratch, so the harness still proves the engine is a
//! function of its inputs.
//!
//! Orderings an API does not promise are normalised before comparison; values never
//! are, except fields a store stamps from the wall clock, which are compared as
//! present-or-absent because no two runs can agree on them.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::{Debug, Display};
use std::sync::Arc;

use chrono::{DateTime, Duration, TimeZone, Utc};
use proptest::prelude::*;
use proptest::test_runner::{RngAlgorithm, RngSeed};
use serde_json::Value;
use tempfile::TempDir;
use uuid::Uuid;

use crate::config::StorageBackend;
use crate::graph_memory::{
    EntityLabel, EntityNode, EpisodeSource, EpisodicNode, PiiClassification, RelationType,
    RelationshipEdge,
};
use crate::memory::context::ContextBuilder;
use crate::memory::introspection::InterferenceType;
use crate::memory::replay::InterferenceRecord;
use crate::memory::storage::{Modality, ModalityVectors, SearchCriteria, VectorMappingEntry};
use crate::memory::types::{EpisodeContext, Experience, ExperienceType, MemoryTier};
use crate::memory::{Memory, MemoryId};
use crate::storage::test_support::{self, BatchWriterUnderTest, NamedStore};
use crate::storage::{AuditLogEntry, AuditStore, GraphStore, KeyValueStore, PrimaryMemoryStore};
use crate::storage::{ColumnStore, WriteBatch};

// =============================================================================
// Run configuration
// =============================================================================

/// Fixed so CI replays the same sequences on every run; a failure is reproduced by
/// re-running the test, and proptest's regression file pins the shrunk case.
const DIFFERENTIAL_SEED: u64 = 0x0007_D1FF_E2E2_0007;
const CASES: u32 = 64;
const MAX_OPS: usize = 60;

fn harness_config() -> ProptestConfig {
    ProptestConfig {
        cases: CASES,
        max_shrink_iters: 256,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(DIFFERENTIAL_SEED),
        ..ProptestConfig::default()
    }
}

// =============================================================================
// The allow-list
// =============================================================================

/// A behaviour the engines still define differently. The generator stays out of its
/// domain, as `why` says, so a difference the harness observes is always a defect.
struct KnownDivergence {
    name: &'static str,
    rocksdb: &'static str,
    redb: &'static str,
    why: &'static str,
}

const KNOWN_DIVERGENCES: [&KnownDivergence; 0] = [];

// =============================================================================
// Subjects, answers and comparison
// =============================================================================

struct Subject<T> {
    label: String,
    store: T,
    _dir: TempDir,
}

fn label_subjects<T>(
    built: Vec<(StorageBackend, T, TempDir)>,
    instance: &str,
    implementation: &impl Fn(&T) -> Option<&'static str>,
) -> Vec<Subject<T>> {
    built
        .into_iter()
        .map(|(backend, store, dir)| {
            let label = match implementation(&store) {
                Some(name) => format!("{}/{name}{instance}", backend.as_str()),
                None => format!("{}{instance}", backend.as_str()),
            };
            Subject {
                label,
                store,
                _dir: dir,
            }
        })
        .collect()
}

#[cfg(all(feature = "rocksdb", feature = "storage-redb"))]
fn subjects<T>(
    build: impl Fn() -> Vec<(StorageBackend, T, TempDir)>,
    implementation: impl Fn(&T) -> Option<&'static str>,
) -> Vec<Subject<T>> {
    let built = build();
    test_support::assert_both_backends(&built);
    label_subjects(built, "", &implementation)
}

#[cfg(not(all(feature = "rocksdb", feature = "storage-redb")))]
fn subjects<T>(
    build: impl Fn() -> Vec<(StorageBackend, T, TempDir)>,
    implementation: impl Fn(&T) -> Option<&'static str>,
) -> Vec<Subject<T>> {
    let mut subjects = label_subjects(build(), "", &implementation);
    subjects.extend(label_subjects(build(), " (replay)", &implementation));
    subjects
}

/// A store's answer to one call. A refusal is compared as a refusal and its message
/// dropped: the engines word the same refusal differently ("Memory not found:
/// MemoryId(..)" against "Memory not found: <uuid>").
#[derive(Debug, Clone, PartialEq)]
enum Answer<T> {
    Value(T),
    Refused,
}

impl<T> Answer<T> {
    fn of<U>(result: anyhow::Result<U>, view: impl FnOnce(U) -> T) -> Self {
        match result {
            Ok(value) => Answer::Value(view(value)),
            Err(_) => Answer::Refused,
        }
    }
}

/// What a mutating call returned.
#[derive(Debug, Clone, PartialEq)]
enum Outcome {
    Done,
    Count(usize),
    Ids(Vec<MemoryId>),
    Migration(usize, usize, usize),
    Refused,
}

fn done(result: anyhow::Result<()>) -> Outcome {
    match result {
        Ok(()) => Outcome::Done,
        Err(_) => Outcome::Refused,
    }
}

fn counted(result: anyhow::Result<usize>) -> Outcome {
    match result {
        Ok(count) => Outcome::Count(count),
        Err(_) => Outcome::Refused,
    }
}

#[derive(Default)]
struct Comparison {
    mismatches: Vec<String>,
}

impl Comparison {
    fn equal<T: PartialEq + Debug>(&mut self, what: impl Display, left: &T, right: &T) {
        if left != right {
            self.mismatches
                .push(format!("{what}:\n      {left:?}\n   vs {right:?}"));
        }
    }
}

/// Every subject's snapshot against the first's.
fn agree<T, S>(
    subjects: &[Subject<T>],
    snapshots: &[S],
    context: &str,
    compare: impl Fn(&mut Comparison, &S, &S),
) -> Result<(), TestCaseError> {
    for (subject, snapshot) in subjects.iter().zip(snapshots).skip(1) {
        let mut comparison = Comparison::default();
        compare(&mut comparison, &snapshots[0], snapshot);
        if !comparison.mismatches.is_empty() {
            return Err(TestCaseError::fail(format!(
                "{context}: {} and {} disagree\n  {}",
                subjects[0].label,
                subject.label,
                comparison.mismatches.join("\n  ")
            )));
        }
    }
    Ok(())
}

/// Every subject's outcome against the first's.
fn agree_on_outcomes<T>(
    subjects: &[Subject<T>],
    outcomes: &[Outcome],
    context: &str,
) -> Result<(), TestCaseError> {
    let reference = &outcomes[0];
    for (subject, outcome) in subjects.iter().zip(outcomes).skip(1) {
        if outcome != reference {
            return Err(TestCaseError::fail(format!(
                "{context}: {} returned {reference:?}, {} returned {outcome:?}",
                subjects[0].label, subject.label
            )));
        }
    }
    Ok(())
}

fn json<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).expect("every stored record serializes to JSON")
}

// =============================================================================
// Shared time and value domain
// =============================================================================

/// Every generated timestamp falls within days of this; anything at or after
/// [`clock_floor`] was stamped by a store from the wall clock.
fn epoch() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2002, 3, 4, 0, 0, 0)
        .single()
        .expect("a valid instant")
}

fn clock_floor() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2010, 1, 1, 0, 0, 0)
        .single()
        .expect("a valid instant")
}

const CLOCK_STAMP: &str = "<stamped from the wall clock>";

fn day_start(day: i64) -> DateTime<Utc> {
    epoch() + Duration::days(day)
}

fn value_bytes() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..4)
}

#[derive(Debug, Clone)]
enum KvOp {
    Put { key: usize, value: Vec<u8> },
    Delete { key: usize },
    PutMany(Vec<(usize, Vec<u8>)>),
    DeleteMany(Vec<usize>),
    Flush,
}

fn kv_op(key_count: usize) -> impl Strategy<Value = KvOp> {
    prop_oneof![
        5 => (0..key_count, value_bytes()).prop_map(|(key, value)| KvOp::Put { key, value }),
        3 => (0..key_count).prop_map(|key| KvOp::Delete { key }),
        2 => prop::collection::vec((0..key_count, value_bytes()), 0..4).prop_map(KvOp::PutMany),
        1 => prop::collection::vec(0..key_count, 0..4).prop_map(KvOp::DeleteMany),
        1 => Just(KvOp::Flush),
    ]
}

#[derive(Debug, Clone)]
enum KvAction {
    Put(Vec<u8>, Vec<u8>),
    Delete(Vec<u8>),
    PutMany(Vec<(Vec<u8>, Vec<u8>)>),
    DeleteMany(Vec<Vec<u8>>),
    Flush,
}

impl KvAction {
    fn resolve(op: &KvOp, keys: &[&[u8]]) -> Self {
        let key = |index: usize| keys[index % keys.len()].to_vec();
        match op {
            KvOp::Put { key: k, value } => KvAction::Put(key(*k), value.clone()),
            KvOp::Delete { key: k } => KvAction::Delete(key(*k)),
            KvOp::PutMany(pairs) => {
                KvAction::PutMany(pairs.iter().map(|(k, v)| (key(*k), v.clone())).collect())
            }
            KvOp::DeleteMany(indices) => {
                KvAction::DeleteMany(indices.iter().map(|k| key(*k)).collect())
            }
            KvOp::Flush => KvAction::Flush,
        }
    }
}

fn apply_kv(store: &dyn KeyValueStore, action: &KvAction) -> Outcome {
    match action {
        KvAction::Put(key, value) => done(store.put_bytes(key, value)),
        KvAction::Delete(key) => done(store.delete_bytes(key)),
        KvAction::PutMany(pairs) => done(store.put_many(pairs)),
        KvAction::DeleteMany(keys) => done(store.delete_many(keys)),
        KvAction::Flush => done(store.flush()),
    }
}

type Rows = Vec<(Vec<u8>, Vec<u8>)>;

const SCAN_LIMITS: [Option<usize>; 4] = [None, Some(0), Some(1), Some(3)];

// =============================================================================
// PrimaryMemoryStore
// =============================================================================

const SLOTS: usize = 8;
const CONTENT_WORDS: [&str; 3] = ["amber", "basalt", "cobalt"];
const CONTENT_GROUPS: usize = 3;
const TAGS: [&str; 4] = ["alpha", "ALPHA", "beta", "Gamma"];
const ENTITIES: [&str; 4] = ["Rust", "rust", "redb", "Tokio"];
const EPISODES: [&str; 2] = ["ep-a", "ep-b"];
const ROBOTS: [&str; 2] = ["rover-1", "rover-2"];
const MISSIONS: [&str; 2] = ["north", "south"];
const ACTIONS: [&str; 2] = ["grasp", "navigate"];
const TIERS: [MemoryTier; 4] = [
    MemoryTier::Working,
    MemoryTier::Session,
    MemoryTier::LongTerm,
    MemoryTier::Archive,
];
const EXPERIENCE_TYPE_COUNT: usize = 5;
const EXTERNAL_VARIANTS: usize = 2;
/// Clusters a search radius cuts through: London with sites about 50 km east and north
/// of it, either side of the antimeridian, and one degree off the north pole, where
/// RocksDB's geohash cells are narrowest and its candidate grid is widest.
const SITES: [(f64, f64); 8] = [
    (51.5074, -0.1278),
    (51.5074, 0.5900),
    (51.9500, -0.1278),
    (48.8566, 2.3522),
    (40.4168, -3.7038),
    (10.0, 179.95),
    (10.0, -179.95),
    (89.0, 45.0),
];
const OPEN_OCEAN: (f64, f64) = (-30.0, -140.0);
const SEARCH_RADIUS_METRES: f64 = 50_000.0;
/// Importance in fortieths, reward in fortieths of its [-1, 1] span: every RocksDB
/// bucket edge (tenths, twentieths) and the quarter points between them are drawn.
const IMPORTANCE_STEPS: u8 = 40;
const REWARD_STEPS: u8 = 80;
/// Second-of-day values on the boundaries the date probes use.
const DAY_MARKS: [i64; 5] = [0, 21_600, 43_200, 64_800, 86_399];
const INTERFERENCE_HOLDERS: [&str; 3] = ["holder-0", "holder-1", "holder-2"];
const WATERMARK_USERS: [&str; 2] = ["alice", "bob"];
const MERGE_USER: &str = "differential-user";
const KV_KEYS: [&[u8]; 6] = [b"dials", b"dials:a", b"dials:b", b"e", b"\x00", b"\xff\x01"];
/// Never 16 bytes long: rocksdb shares its default family with memories and takes
/// every 16-byte key there for a memory id.
const DEFAULT_KV_KEYS: [&[u8]; 4] = [b"dkv:a", b"dkv:ab", b"dkv:b", b"dkv:zz"];
const DEFAULT_KV_PREFIXES: [&[u8]; 4] = [b"dkv:", b"interference", b"stats:", b"_watermark:"];
const FOREIGN_KEY_PREFIXES: [&[u8]; 3] = [b"", b"dkv:", b"vmapping:"];
const UNDECLARED_FAMILY: &str = "undeclared_family";

/// Slot ids open with a byte no ASCII key starts with, so a one-byte prefix scan in
/// memory-id space reaches memories and nothing else on either engine.
fn slot_id(slot: usize) -> MemoryId {
    let mut bytes = [0u8; 16];
    bytes[0] = 0xA0 + slot as u8;
    bytes[1] = 0x5E;
    for (index, byte) in bytes.iter_mut().enumerate().skip(2) {
        *byte = (slot as u8).wrapping_mul(31).wrapping_add(index as u8);
    }
    MemoryId(Uuid::from_bytes(bytes))
}

/// Shared by the slots of one group, so two live memories often hold the same content
/// and a content-hash lookup has to choose between them the same way on every engine.
fn content(slot: usize, word: usize) -> String {
    format!(
        "group {} remembers {}",
        slot % CONTENT_GROUPS,
        CONTENT_WORDS[word]
    )
}

fn external_id(slot: usize, variant: usize) -> String {
    format!("ext:{slot}:{variant}")
}

fn experience_type(index: usize) -> ExperienceType {
    match index % EXPERIENCE_TYPE_COUNT {
        0 => ExperienceType::Observation,
        1 => ExperienceType::Task,
        2 => ExperienceType::Decision,
        3 => ExperienceType::Learning,
        _ => ExperienceType::Error,
    }
}

fn importance_at(step: u8) -> f32 {
    f32::from(step) / f32::from(IMPORTANCE_STEPS)
}

fn reward_at(step: u8) -> f32 {
    2.0 * f32::from(step) / f32::from(REWARD_STEPS) - 1.0
}

fn importance_bounds(first_step: u8, last_step: u8) -> SearchCriteria {
    SearchCriteria::ByImportance {
        min: importance_at(first_step),
        max: importance_at(last_step),
    }
}

fn reward_bounds(first_step: u8, last_step: u8) -> SearchCriteria {
    SearchCriteria::ByReward {
        min: reward_at(first_step),
        max: reward_at(last_step),
    }
}

fn instant(day: i64, second_of_day: i64) -> DateTime<Utc> {
    day_start(day) + Duration::seconds(second_of_day)
}

fn date_bounds(start: DateTime<Utc>, end: DateTime<Utc>) -> SearchCriteria {
    SearchCriteria::ByDate { start, end }
}

fn whole_days(first_day: i64, last_day: i64) -> SearchCriteria {
    date_bounds(
        day_start(first_day),
        day_start(last_day + 1) - Duration::nanoseconds(1),
    )
}

fn within(site: (f64, f64), radius_meters: f64) -> SearchCriteria {
    SearchCriteria::ByLocation {
        lat: site.0,
        lon: site.1,
        radius_meters,
    }
}

fn near(site: (f64, f64)) -> SearchCriteria {
    within(site, SEARCH_RADIUS_METRES)
}

#[derive(Debug, Clone)]
struct MemoryShape {
    kind: usize,
    word: usize,
    tags: Vec<usize>,
    entities: Vec<usize>,
    episode: Option<(usize, Option<u32>)>,
    robot: Option<usize>,
    mission: Option<usize>,
    site: Option<usize>,
    action: Option<usize>,
    reward_step: Option<u8>,
}

#[derive(Debug, Clone)]
struct MemoryLifecycle {
    importance_step: u8,
    day: i64,
    second_of_day: i64,
    accessed_after_seconds: i64,
    tier: usize,
    access_count: u32,
    compressed: bool,
    external: Option<usize>,
    forgotten: bool,
}

#[derive(Debug, Clone)]
struct MemoryLinks {
    parent: Option<usize>,
    related: Vec<usize>,
    causal: Vec<usize>,
    embedding: Option<Vec<i8>>,
}

#[derive(Debug, Clone)]
struct MemorySpec {
    shape: MemoryShape,
    lifecycle: MemoryLifecycle,
    links: MemoryLinks,
}

prop_compose! {
    fn memory_shape()(
        kind in 0..EXPERIENCE_TYPE_COUNT,
        word in 0..CONTENT_WORDS.len(),
        tags in prop::collection::vec(0..TAGS.len(), 0..3),
        entities in prop::collection::vec(0..ENTITIES.len(), 0..3),
        episode in prop::option::of((0..EPISODES.len(), prop::option::of(0u32..4))),
        robot in prop::option::of(0..ROBOTS.len()),
        mission in prop::option::of(0..MISSIONS.len()),
        site in prop::option::of(0..SITES.len()),
        action in prop::option::of(0..ACTIONS.len()),
        reward_step in prop::option::of(0u8..=REWARD_STEPS),
    ) -> MemoryShape {
        MemoryShape { kind, word, tags, entities, episode, robot, mission, site, action, reward_step }
    }
}

prop_compose! {
    fn memory_lifecycle()(
        importance_step in 0u8..=IMPORTANCE_STEPS,
        day in 0i64..7,
        second_of_day in prop_oneof![3 => 0i64..86_400, 1 => prop::sample::select(DAY_MARKS.to_vec())],
        accessed_after_seconds in 0i64..86_400,
        tier in 0..TIERS.len(),
        access_count in 0u32..3,
        compressed in any::<bool>(),
        external in prop::option::of(0..EXTERNAL_VARIANTS),
        forgotten in prop::bool::weighted(0.15),
    ) -> MemoryLifecycle {
        MemoryLifecycle {
            importance_step,
            day,
            second_of_day,
            accessed_after_seconds,
            tier,
            access_count,
            compressed,
            external,
            forgotten,
        }
    }
}

prop_compose! {
    fn memory_links()(
        parent in prop::option::of(0..SLOTS),
        related in prop::collection::vec(0..SLOTS, 0..3),
        causal in prop::collection::vec(0..SLOTS, 0..2),
        embedding in prop::option::of(prop::collection::vec(any::<i8>(), 8)),
    ) -> MemoryLinks {
        MemoryLinks { parent, related, causal, embedding }
    }
}

fn memory_spec() -> impl Strategy<Value = Box<MemorySpec>> {
    (memory_shape(), memory_lifecycle(), memory_links()).prop_map(|(shape, lifecycle, links)| {
        Box::new(MemorySpec {
            shape,
            lifecycle,
            links,
        })
    })
}

fn build_memory(slot: usize, spec: &MemorySpec) -> Memory {
    let MemorySpec {
        shape,
        lifecycle,
        links,
    } = spec;
    let created_at = instant(lifecycle.day, lifecycle.second_of_day);
    let mut metadata = HashMap::new();
    metadata.insert("origin".to_string(), CONTENT_WORDS[shape.word].to_string());
    if lifecycle.forgotten {
        metadata.insert("forgotten".to_string(), "true".to_string());
    }
    let context = shape.episode.map(|(episode, sequence_number)| {
        ContextBuilder::new()
            .with_episode(EpisodeContext {
                episode_id: Some(EPISODES[episode].to_string()),
                sequence_number,
                ..EpisodeContext::default()
            })
            .build()
    });
    let experience = Experience {
        experience_type: experience_type(shape.kind),
        content: content(slot, shape.word),
        context,
        entities: shape
            .entities
            .iter()
            .map(|index| ENTITIES[*index].to_string())
            .collect(),
        metadata,
        embeddings: links
            .embedding
            .as_ref()
            .map(|raw| raw.iter().map(|value| f32::from(*value) / 16.0).collect()),
        related_memories: links.related.iter().map(|other| slot_id(*other)).collect(),
        causal_chain: links.causal.iter().map(|other| slot_id(*other)).collect(),
        robot_id: shape.robot.map(|index| ROBOTS[index].to_string()),
        mission_id: shape.mission.map(|index| MISSIONS[index].to_string()),
        geo_location: shape
            .site
            .map(|index| [SITES[index].0, SITES[index].1, 0.0]),
        action_type: shape.action.map(|index| ACTIONS[index].to_string()),
        reward: shape.reward_step.map(reward_at),
        tags: shape
            .tags
            .iter()
            .map(|index| TAGS[*index].to_string())
            .collect(),
        ..Experience::default()
    };
    let mut memory = Memory::from_legacy(
        slot_id(slot),
        experience,
        importance_at(lifecycle.importance_step),
        lifecycle.access_count,
        created_at,
        created_at + Duration::seconds(lifecycle.accessed_after_seconds),
        lifecycle.compressed,
        TIERS[lifecycle.tier],
        Vec::new(),
        1.0,
        None,
        None,
        None,
        None,
        1.0,
        None,
        lifecycle.external.map(|variant| external_id(slot, variant)),
        1,
        Vec::new(),
        Vec::new(),
    );
    // A parent strictly below the slot keeps every hierarchy acyclic, so ancestor
    // walks stay short; both engines carry the identical 100-step cycle bound.
    memory.parent_id = links.parent.filter(|parent| *parent < slot).map(slot_id);
    memory
}

#[derive(Debug, Clone)]
struct MappingSpec {
    text: Vec<u32>,
    image: Option<Vec<u32>>,
    chunked: bool,
    created_offset_millis: i64,
}

prop_compose! {
    fn mapping_spec()(
        text in prop::collection::vec(0u32..64, 0..3),
        image in prop::option::of(prop::collection::vec(0u32..64, 1..3)),
        chunked in any::<bool>(),
        created_offset_millis in 0i64..1_000_000,
    ) -> MappingSpec {
        MappingSpec { text, image, chunked, created_offset_millis }
    }
}

fn build_mapping(spec: &MappingSpec) -> VectorMappingEntry {
    let mut modalities = HashMap::new();
    modalities.insert(
        Modality::Text,
        ModalityVectors {
            vector_ids: spec.text.clone(),
            dimension: Modality::Text.dimension(),
            chunk_ranges: spec.chunked.then(|| vec![(0, 4), (4, 9)]),
        },
    );
    if let Some(image) = &spec.image {
        modalities.insert(
            Modality::Image,
            ModalityVectors {
                vector_ids: image.clone(),
                dimension: Modality::Image.dimension(),
                chunk_ranges: None,
            },
        );
    }
    VectorMappingEntry {
        modalities,
        created_at: epoch().timestamp_millis() + spec.created_offset_millis,
        version: 1,
    }
}

#[derive(Debug, Clone)]
struct InterferenceSpec {
    interfering: usize,
    similarity_percent: u8,
    kind: usize,
    strength_change_percent: i8,
    second: i64,
}

prop_compose! {
    fn interference_spec()(
        interfering in 0..SLOTS,
        similarity_percent in 0u8..=100,
        kind in 0usize..3,
        strength_change_percent in any::<i8>(),
        second in 0i64..86_400,
    ) -> InterferenceSpec {
        InterferenceSpec { interfering, similarity_percent, kind, strength_change_percent, second }
    }
}

fn build_interference(spec: &InterferenceSpec) -> InterferenceRecord {
    InterferenceRecord {
        interfering_memory_id: slot_id(spec.interfering).0.to_string(),
        similarity: f32::from(spec.similarity_percent) / 100.0,
        interference_type: match spec.kind {
            0 => InterferenceType::Retroactive,
            1 => InterferenceType::Proactive,
            _ => InterferenceType::RetrievalCompetition,
        },
        strength_change: f32::from(spec.strength_change_percent) / 100.0,
        timestamp: epoch() + Duration::seconds(spec.second),
    }
}

#[derive(Debug, Clone, Copy)]
enum InsertPath {
    Store,
    PutMemory,
    StoreWithVectors,
}

#[derive(Debug, Clone, Copy)]
enum RewritePath {
    Update,
    UpdateMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemovePath {
    Delete,
    DeleteMemory,
}

#[derive(Debug, Clone, Copy)]
enum PrimaryBatchTarget {
    DefaultUnnamed,
    DefaultNamed,
    Kv,
    Undeclared,
}

#[derive(Debug, Clone)]
struct PrimaryBatchEntry {
    target: PrimaryBatchTarget,
    key: usize,
    value: Option<Vec<u8>>,
}

fn primary_batch_entry() -> impl Strategy<Value = PrimaryBatchEntry> {
    let target = prop_oneof![
        3 => Just(PrimaryBatchTarget::DefaultUnnamed),
        2 => Just(PrimaryBatchTarget::DefaultNamed),
        2 => Just(PrimaryBatchTarget::Kv),
        1 => Just(PrimaryBatchTarget::Undeclared),
    ];
    (target, 0usize..8, prop::option::of(value_bytes()))
        .prop_map(|(target, key, value)| PrimaryBatchEntry { target, key, value })
}

#[derive(Debug, Clone)]
enum PrimaryOp {
    Insert {
        slot: usize,
        spec: Box<MemorySpec>,
        path: InsertPath,
        vector_ids: Vec<u32>,
    },
    Rewrite {
        slot: usize,
        spec: Box<MemorySpec>,
        path: RewritePath,
    },
    Remove {
        slot: usize,
        path: RemovePath,
    },
    UpdateAccess {
        slot: usize,
    },
    PutVectorMapping {
        slot: usize,
        mapping: MappingSpec,
    },
    UpdateVectorMapping {
        slot: usize,
        vector_ids: Vec<u32>,
    },
    DeleteVectorMapping {
        slot: usize,
    },
    MarkForgottenByAge {
        day: i64,
    },
    MarkForgottenByImportance {
        step: u8,
    },
    IncrementRetrievalCount,
    SaveInterference {
        holder: usize,
        records: Vec<InterferenceSpec>,
    },
    DeleteInterference {
        holder: usize,
    },
    ClearInterference,
    SaveInterferenceEventCount {
        count: usize,
    },
    SetFactWatermark {
        user: usize,
        millis: i64,
    },
    Kv(KvOp),
    DefaultKv(KvOp),
    WriteBatch(Vec<PrimaryBatchEntry>),
    Flush,
    CleanupCorrupted,
    MigrateLegacy,
    RunMergePass {
        limit: usize,
    },
    GcMergeTombstones,
}

fn primary_op() -> impl Strategy<Value = PrimaryOp> {
    let insert_path = prop_oneof![
        Just(InsertPath::Store),
        Just(InsertPath::PutMemory),
        Just(InsertPath::StoreWithVectors),
    ];
    let rewrite_path = prop_oneof![Just(RewritePath::Update), Just(RewritePath::UpdateMemory)];
    let remove_path = prop_oneof![Just(RemovePath::Delete), Just(RemovePath::DeleteMemory)];
    prop_oneof![
        10 => (0..SLOTS, memory_spec(), insert_path, prop::collection::vec(0u32..64, 0..3))
            .prop_map(|(slot, spec, path, vector_ids)| PrimaryOp::Insert { slot, spec, path, vector_ids }),
        5 => (0..SLOTS, memory_spec(), rewrite_path)
            .prop_map(|(slot, spec, path)| PrimaryOp::Rewrite { slot, spec, path }),
        3 => (0..SLOTS, remove_path).prop_map(|(slot, path)| PrimaryOp::Remove { slot, path }),
        3 => (0..SLOTS).prop_map(|slot| PrimaryOp::UpdateAccess { slot }),
        2 => (0..SLOTS, mapping_spec())
            .prop_map(|(slot, mapping)| PrimaryOp::PutVectorMapping { slot, mapping }),
        2 => (0..SLOTS, prop::collection::vec(0u32..64, 0..3))
            .prop_map(|(slot, vector_ids)| PrimaryOp::UpdateVectorMapping { slot, vector_ids }),
        1 => (0..SLOTS).prop_map(|slot| PrimaryOp::DeleteVectorMapping { slot }),
        1 => (0i64..8).prop_map(|day| PrimaryOp::MarkForgottenByAge { day }),
        1 => (0u8..=IMPORTANCE_STEPS).prop_map(|step| PrimaryOp::MarkForgottenByImportance { step }),
        1 => Just(PrimaryOp::IncrementRetrievalCount),
        2 => (0..INTERFERENCE_HOLDERS.len(), prop::collection::vec(interference_spec(), 0..3))
            .prop_map(|(holder, records)| PrimaryOp::SaveInterference { holder, records }),
        1 => (0..INTERFERENCE_HOLDERS.len()).prop_map(|holder| PrimaryOp::DeleteInterference { holder }),
        1 => Just(PrimaryOp::ClearInterference),
        1 => (0usize..20).prop_map(|count| PrimaryOp::SaveInterferenceEventCount { count }),
        1 => (0..WATERMARK_USERS.len(), any::<i64>())
            .prop_map(|(user, millis)| PrimaryOp::SetFactWatermark { user, millis }),
        2 => kv_op(KV_KEYS.len()).prop_map(PrimaryOp::Kv),
        2 => kv_op(DEFAULT_KV_KEYS.len()).prop_map(PrimaryOp::DefaultKv),
        2 => prop::collection::vec(primary_batch_entry(), 0..4).prop_map(PrimaryOp::WriteBatch),
        1 => Just(PrimaryOp::Flush),
        1 => Just(PrimaryOp::CleanupCorrupted),
        1 => Just(PrimaryOp::MigrateLegacy),
        1 => (0usize..4).prop_map(|limit| PrimaryOp::RunMergePass { limit }),
        1 => Just(PrimaryOp::GcMergeTombstones),
    ]
}

fn primary_ops() -> impl Strategy<Value = Vec<PrimaryOp>> {
    prop::collection::vec(primary_op(), 1..=MAX_OPS)
}

/// A generated op with every choice resolved, applied verbatim to each subject.
enum PrimaryAction {
    Store(Box<Memory>),
    PutMemory(Box<Memory>),
    StoreWithVectors(Box<Memory>, Vec<u32>),
    Update(Box<Memory>),
    UpdateMemory(Box<Memory>),
    Delete(MemoryId),
    DeleteMemory(MemoryId),
    UpdateAccess(MemoryId),
    PutVectorMapping(MemoryId, VectorMappingEntry),
    UpdateVectorMapping(MemoryId, Vec<u32>),
    DeleteVectorMapping(MemoryId),
    MarkForgottenByAge(DateTime<Utc>),
    MarkForgottenByImportance(f32),
    IncrementRetrievalCount,
    SaveInterference(String, Vec<InterferenceRecord>),
    DeleteInterference(String),
    ClearInterference,
    SaveInterferenceEventCount(usize),
    SetFactWatermark(String, i64),
    Kv(KvAction),
    DefaultKv(KvAction),
    WriteBatch(WriteBatch),
    Flush,
    CleanupCorrupted,
    MigrateLegacy,
    RunMergePass(usize),
    GcMergeTombstones,
}

impl PrimaryAction {
    fn describe(&self) -> String {
        match self {
            PrimaryAction::Store(memory) => format!("store {}", memory.id.0),
            PrimaryAction::PutMemory(memory) => format!("put_memory {}", memory.id.0),
            PrimaryAction::StoreWithVectors(memory, ids) => {
                format!("store_with_vectors {} {ids:?}", memory.id.0)
            }
            PrimaryAction::Update(memory) => format!("update {}", memory.id.0),
            PrimaryAction::UpdateMemory(memory) => format!("update_memory {}", memory.id.0),
            PrimaryAction::Delete(id) => format!("delete {}", id.0),
            PrimaryAction::DeleteMemory(id) => format!("delete_memory {}", id.0),
            PrimaryAction::UpdateAccess(id) => format!("update_access {}", id.0),
            PrimaryAction::PutVectorMapping(id, _) => format!("put_vector_mapping {}", id.0),
            PrimaryAction::UpdateVectorMapping(id, ids) => {
                format!("update_vector_mapping {} {ids:?}", id.0)
            }
            PrimaryAction::DeleteVectorMapping(id) => format!("delete_vector_mapping {}", id.0),
            PrimaryAction::MarkForgottenByAge(cutoff) => format!("mark_forgotten_by_age {cutoff}"),
            PrimaryAction::MarkForgottenByImportance(threshold) => {
                format!("mark_forgotten_by_importance {threshold}")
            }
            PrimaryAction::IncrementRetrievalCount => "increment_retrieval_count".to_string(),
            PrimaryAction::SaveInterference(holder, records) => {
                format!(
                    "save_interference_records {holder} ({} records)",
                    records.len()
                )
            }
            PrimaryAction::DeleteInterference(holder) => {
                format!("delete_interference_records {holder}")
            }
            PrimaryAction::ClearInterference => "clear_all_interference_records".to_string(),
            PrimaryAction::SaveInterferenceEventCount(count) => {
                format!("save_interference_event_count {count}")
            }
            PrimaryAction::SetFactWatermark(user, millis) => {
                format!("set_fact_watermark {user} {millis}")
            }
            PrimaryAction::Kv(action) => format!("key_value_store {action:?}"),
            PrimaryAction::DefaultKv(action) => format!("default_key_value_store {action:?}"),
            PrimaryAction::WriteBatch(batch) => format!("write_batch {:?}", batch.ops()),
            PrimaryAction::Flush => "flush".to_string(),
            PrimaryAction::CleanupCorrupted => "cleanup_corrupted".to_string(),
            PrimaryAction::MigrateLegacy => "migrate_legacy".to_string(),
            PrimaryAction::RunMergePass(limit) => format!("run_merge_pass {limit}"),
            PrimaryAction::GcMergeTombstones => "gc_merge_tombstones".to_string(),
        }
    }
}

/// A generated op with every choice resolved. Every op is applied: no slot state is
/// avoided, as the engines define every one of them identically.
fn resolve_primary(op: &PrimaryOp) -> PrimaryAction {
    match op {
        PrimaryOp::Insert {
            slot,
            spec,
            path,
            vector_ids,
        } => {
            let memory = Box::new(build_memory(*slot, spec));
            match path {
                InsertPath::Store => PrimaryAction::Store(memory),
                InsertPath::PutMemory => PrimaryAction::PutMemory(memory),
                InsertPath::StoreWithVectors => {
                    PrimaryAction::StoreWithVectors(memory, vector_ids.clone())
                }
            }
        }
        PrimaryOp::Rewrite { slot, spec, path } => {
            let memory = Box::new(build_memory(*slot, spec));
            match path {
                RewritePath::Update => PrimaryAction::Update(memory),
                RewritePath::UpdateMemory => PrimaryAction::UpdateMemory(memory),
            }
        }
        PrimaryOp::Remove { slot, path } => match path {
            RemovePath::Delete => PrimaryAction::Delete(slot_id(*slot)),
            RemovePath::DeleteMemory => PrimaryAction::DeleteMemory(slot_id(*slot)),
        },
        PrimaryOp::UpdateAccess { slot } => PrimaryAction::UpdateAccess(slot_id(*slot)),
        PrimaryOp::PutVectorMapping { slot, mapping } => {
            PrimaryAction::PutVectorMapping(slot_id(*slot), build_mapping(mapping))
        }
        PrimaryOp::UpdateVectorMapping { slot, vector_ids } => {
            PrimaryAction::UpdateVectorMapping(slot_id(*slot), vector_ids.clone())
        }
        PrimaryOp::DeleteVectorMapping { slot } => {
            PrimaryAction::DeleteVectorMapping(slot_id(*slot))
        }
        PrimaryOp::MarkForgottenByAge { day } => PrimaryAction::MarkForgottenByAge(day_start(*day)),
        PrimaryOp::MarkForgottenByImportance { step } => {
            PrimaryAction::MarkForgottenByImportance(importance_at(*step))
        }
        PrimaryOp::IncrementRetrievalCount => PrimaryAction::IncrementRetrievalCount,
        PrimaryOp::SaveInterference { holder, records } => PrimaryAction::SaveInterference(
            INTERFERENCE_HOLDERS[*holder].to_string(),
            records.iter().map(build_interference).collect(),
        ),
        PrimaryOp::DeleteInterference { holder } => {
            PrimaryAction::DeleteInterference(INTERFERENCE_HOLDERS[*holder].to_string())
        }
        PrimaryOp::ClearInterference => PrimaryAction::ClearInterference,
        PrimaryOp::SaveInterferenceEventCount { count } => {
            PrimaryAction::SaveInterferenceEventCount(*count)
        }
        PrimaryOp::SetFactWatermark { user, millis } => {
            PrimaryAction::SetFactWatermark(WATERMARK_USERS[*user].to_string(), *millis)
        }
        PrimaryOp::Kv(op) => PrimaryAction::Kv(KvAction::resolve(op, &KV_KEYS)),
        PrimaryOp::DefaultKv(op) => {
            PrimaryAction::DefaultKv(KvAction::resolve(op, &DEFAULT_KV_KEYS))
        }
        PrimaryOp::WriteBatch(entries) => primary_batch(entries),
        PrimaryOp::Flush => PrimaryAction::Flush,
        PrimaryOp::CleanupCorrupted => PrimaryAction::CleanupCorrupted,
        PrimaryOp::MigrateLegacy => PrimaryAction::MigrateLegacy,
        PrimaryOp::RunMergePass { limit } => PrimaryAction::RunMergePass(*limit),
        PrimaryOp::GcMergeTombstones => PrimaryAction::GcMergeTombstones,
    }
}

fn primary_batch(entries: &[PrimaryBatchEntry]) -> PrimaryAction {
    let mut batch = WriteBatch::new();
    for entry in entries {
        let (family, key) = match entry.target {
            PrimaryBatchTarget::DefaultUnnamed => {
                (None, DEFAULT_KV_KEYS[entry.key % DEFAULT_KV_KEYS.len()])
            }
            PrimaryBatchTarget::DefaultNamed => (
                Some("default"),
                DEFAULT_KV_KEYS[entry.key % DEFAULT_KV_KEYS.len()],
            ),
            PrimaryBatchTarget::Kv => (Some("kv"), KV_KEYS[entry.key % KV_KEYS.len()]),
            PrimaryBatchTarget::Undeclared => {
                (Some(UNDECLARED_FAMILY), KV_KEYS[entry.key % KV_KEYS.len()])
            }
        };
        match (family, &entry.value) {
            (None, Some(value)) => batch.put(key.to_vec(), value.clone()),
            (None, None) => batch.delete(key.to_vec()),
            (Some(family), Some(value)) => batch.put_cf(family, key.to_vec(), value.clone()),
            (Some(family), None) => batch.delete_cf(family, key.to_vec()),
        }
    }
    PrimaryAction::WriteBatch(batch)
}

fn sorted_ids(memories: Vec<Memory>) -> Vec<MemoryId> {
    let mut ids: Vec<MemoryId> = memories.into_iter().map(|memory| memory.id).collect();
    ids.sort();
    ids
}

fn apply_primary(store: &dyn PrimaryMemoryStore, action: &PrimaryAction) -> Outcome {
    match action {
        PrimaryAction::Store(memory) => done(store.store(memory)),
        PrimaryAction::PutMemory(memory) => done(store.put_memory(memory)),
        PrimaryAction::StoreWithVectors(memory, vector_ids) => {
            done(store.store_with_vectors(memory, vector_ids.clone()))
        }
        PrimaryAction::Update(memory) => done(store.update(memory)),
        PrimaryAction::UpdateMemory(memory) => done(store.update_memory(memory)),
        PrimaryAction::Delete(id) => done(store.delete(id)),
        PrimaryAction::DeleteMemory(id) => done(store.delete_memory(id)),
        PrimaryAction::UpdateAccess(id) => done(store.update_access(id)),
        PrimaryAction::PutVectorMapping(id, mapping) => done(store.put_vector_mapping(id, mapping)),
        PrimaryAction::UpdateVectorMapping(id, vector_ids) => {
            done(store.update_vector_mapping(id, vector_ids.clone()))
        }
        PrimaryAction::DeleteVectorMapping(id) => done(store.delete_vector_mapping(id)),
        PrimaryAction::MarkForgottenByAge(cutoff) => match store.mark_forgotten_by_age(*cutoff) {
            Ok(mut ids) => {
                ids.sort();
                Outcome::Ids(ids)
            }
            Err(_) => Outcome::Refused,
        },
        PrimaryAction::MarkForgottenByImportance(threshold) => {
            match store.mark_forgotten_by_importance(*threshold) {
                Ok(mut ids) => {
                    ids.sort();
                    Outcome::Ids(ids)
                }
                Err(_) => Outcome::Refused,
            }
        }
        PrimaryAction::IncrementRetrievalCount => counted(store.increment_retrieval_count()),
        PrimaryAction::SaveInterference(holder, records) => {
            done(store.save_interference_records(holder, records))
        }
        PrimaryAction::DeleteInterference(holder) => {
            done(store.delete_interference_records(holder))
        }
        PrimaryAction::ClearInterference => counted(store.clear_all_interference_records()),
        PrimaryAction::SaveInterferenceEventCount(count) => {
            done(store.save_interference_event_count(*count))
        }
        PrimaryAction::SetFactWatermark(user, millis) => {
            store.set_fact_watermark(user, *millis);
            Outcome::Done
        }
        PrimaryAction::Kv(action) => apply_kv(store.key_value_store().as_ref(), action),
        PrimaryAction::DefaultKv(action) => {
            apply_kv(store.default_key_value_store().as_ref(), action)
        }
        PrimaryAction::WriteBatch(batch) => done(store.write_batch(batch)),
        PrimaryAction::Flush => done(store.flush()),
        PrimaryAction::CleanupCorrupted => counted(store.cleanup_corrupted()),
        PrimaryAction::MigrateLegacy => match store.migrate_legacy() {
            Ok((migrated, current, failed)) => Outcome::Migration(migrated, current, failed),
            Err(_) => Outcome::Refused,
        },
        PrimaryAction::RunMergePass(limit) => counted(store.run_merge_pass(MERGE_USER, *limit)),
        PrimaryAction::GcMergeTombstones => counted(
            store
                .gc_merge_tombstones(MERGE_USER)
                .map(|reclaimed| reclaimed.len()),
        ),
    }
}

/// A memory as JSON, with the fields a store stamps from the wall clock reduced to
/// "stamped" — update_access sets `last_accessed` and appends to `access_history`, a
/// forget pass sets `forgotten_at` — and the bytes those stamps' text took.
fn stamped_view(memory: &Memory) -> (Value, usize) {
    let mut view = json(memory);
    let mut stamp_bytes = 0;
    if let Some(object) = view.as_object_mut() {
        if let Some(stamp) = object.get_mut("last_accessed") {
            stamp_bytes += redact_clock(stamp);
        }
        if let Some(Value::Array(history)) = object.get_mut("access_history") {
            stamp_bytes += history.iter_mut().map(redact_clock).sum::<usize>();
        }
        if let Some(Value::Object(metadata)) = object
            .get_mut("experience")
            .and_then(|experience| experience.get_mut("metadata"))
        {
            if let Some(stamp) = metadata.get_mut("forgotten_at") {
                stamp_bytes += redact_clock(stamp);
            }
        }
    }
    (view, stamp_bytes)
}

fn memory_view(memory: &Memory) -> Value {
    stamped_view(memory).0
}

/// The length of the text replaced, or 0 when `stamp` was not clock-stamped.
fn redact_clock(stamp: &mut Value) -> usize {
    let stamped_len = stamp
        .as_str()
        .filter(|text| {
            DateTime::parse_from_rfc3339(text)
                .is_ok_and(|instant| instant.with_timezone(&Utc) >= clock_floor())
        })
        .map(str::len);
    match stamped_len {
        Some(len) => {
            *stamp = Value::String(CLOCK_STAMP.to_string());
            len
        }
        None => 0,
    }
}

#[derive(Debug, PartialEq)]
struct MemoryRecord {
    id: MemoryId,
    forgotten: bool,
    view: Value,
}

/// The records sorted by id, and the clock-stamp bytes the unforgotten ones carry.
fn memory_records(memories: &[Memory]) -> (Vec<MemoryRecord>, usize) {
    let mut live_stamp_bytes = 0;
    let mut records: Vec<MemoryRecord> = memories
        .iter()
        .map(|memory| {
            let (view, stamp_bytes) = stamped_view(memory);
            let forgotten = memory.is_forgotten();
            if !forgotten {
                live_stamp_bytes += stamp_bytes;
            }
            MemoryRecord {
                id: memory.id.clone(),
                forgotten,
                view,
            }
        })
        .collect();
    records.sort_by(|a, b| a.id.cmp(&b.id));
    (records, live_stamp_bytes)
}

#[derive(Debug, PartialEq)]
enum Stamp {
    Generated(i64),
    Clock,
}

#[derive(Debug, PartialEq)]
struct ModalityView {
    vector_ids: Vec<u32>,
    dimension: usize,
    chunk_ranges: Option<Vec<(usize, usize)>>,
}

#[derive(Debug, PartialEq)]
struct MappingView {
    modalities: BTreeMap<&'static str, ModalityView>,
    created_at: Stamp,
    version: u8,
}

impl From<&VectorMappingEntry> for MappingView {
    fn from(entry: &VectorMappingEntry) -> Self {
        Self {
            modalities: entry
                .modalities
                .iter()
                .map(|(modality, vectors)| {
                    (
                        modality.as_str(),
                        ModalityView {
                            vector_ids: vectors.vector_ids.clone(),
                            dimension: vectors.dimension,
                            chunk_ranges: vectors.chunk_ranges.clone(),
                        },
                    )
                })
                .collect(),
            // A mapping created by update_vector_mapping or store_with_vectors starts
            // from VectorMappingEntry::default(), which reads the clock.
            created_at: if entry.created_at >= clock_floor().timestamp_millis() {
                Stamp::Clock
            } else {
                Stamp::Generated(entry.created_at)
            },
            version: entry.version,
        }
    }
}

#[derive(Debug, PartialEq)]
struct StatsView {
    total_count: usize,
    compressed_count: usize,
    /// chrono writes 0, 3, 6 or 9 fractional digits depending on the instant, so two
    /// stores stamping the same update_access can hold records of different lengths;
    /// the stamps' own bytes are taken out before the sizes are compared.
    total_size_bytes_net_of_clock_stamps: usize,
    average_importance_bits: u32,
    importance_sum_bits: u32,
    total_retrievals: usize,
}

#[derive(Debug, PartialEq)]
struct SlotObservation {
    get: Answer<Value>,
    get_memory: Answer<Option<Value>>,
    children: Answer<Vec<MemoryId>>,
    ancestors: Answer<Vec<MemoryId>>,
    hierarchy: Answer<(Vec<MemoryId>, MemoryId, Vec<MemoryId>)>,
    vector_mapping: Answer<Option<MappingView>>,
    inbound_refs: Answer<bool>,
}

#[derive(Debug, PartialEq)]
struct PrimarySnapshot {
    all: Answer<Vec<MemoryRecord>>,
    all_ids: Answer<Vec<MemoryId>>,
    slots: Vec<SlotObservation>,
    searches: Vec<Answer<Vec<MemoryId>>>,
    empty_combined: Answer<Vec<MemoryId>>,
    content_hash: Vec<Option<MemoryId>>,
    external: Vec<Answer<Option<Value>>>,
    uncompressed: Vec<Answer<Vec<MemoryId>>>,
    stats: Answer<StatsView>,
    all_mappings: Answer<Vec<(MemoryId, MappingView)>>,
    interference: Answer<(BTreeMap<String, Value>, usize)>,
    watermarks: Vec<Option<i64>>,
    kv: Answer<Rows>,
    default_kv: Vec<Answer<Rows>>,
    memory_prefix_keys: Vec<Answer<Vec<Vec<u8>>>>,
    foreign_prefix_keys: Vec<Answer<Vec<Vec<u8>>>>,
}

/// The fixed reads every snapshot makes: one per search variant, and one per value
/// any generated memory can carry.
struct PrimaryProbes {
    searches: Vec<(&'static str, SearchCriteria)>,
    contents: Vec<String>,
    external_ids: Vec<String>,
    cutoffs: Vec<DateTime<Utc>>,
    memory_prefixes: Vec<Vec<u8>>,
}

impl PrimaryProbes {
    fn new() -> Self {
        use SearchCriteria as By;
        let searches = vec![
            ("ByDate day 0", whole_days(0, 0)),
            ("ByDate days 1-3", whole_days(1, 3)),
            ("ByDate days 4-6", whole_days(4, 6)),
            ("ByDate reversed", whole_days(3, 1)),
            (
                "ByDate day 1 06:00 to day 3 18:00",
                date_bounds(instant(1, DAY_MARKS[1]), instant(3, DAY_MARKS[3])),
            ),
            (
                "ByDate day 2 noon to noon",
                date_bounds(instant(2, DAY_MARKS[2]), instant(2, DAY_MARKS[2])),
            ),
            (
                "ByDate day 4 morning",
                date_bounds(instant(4, DAY_MARKS[0]), instant(4, DAY_MARKS[2])),
            ),
            (
                "ByDate day 5 evening, reversed",
                date_bounds(instant(5, DAY_MARKS[4]), instant(5, DAY_MARKS[3])),
            ),
            (
                "ByType Observation",
                By::ByType(ExperienceType::Observation),
            ),
            ("ByType Task", By::ByType(ExperienceType::Task)),
            ("ByType Intention", By::ByType(ExperienceType::Intention)),
            ("ByImportance 0 to 0.4", importance_bounds(0, 16)),
            ("ByImportance 0.425 to 1", importance_bounds(17, 40)),
            ("ByImportance exactly 0.3", importance_bounds(12, 12)),
            ("ByImportance 0.25 to 0.35", importance_bounds(10, 14)),
            ("ByImportance 0 to 1", importance_bounds(0, 40)),
            ("ByImportance reversed", importance_bounds(30, 10)),
            ("ByEntity rust", By::ByEntity("rust".to_string())),
            ("ByEntity REDB", By::ByEntity("REDB".to_string())),
            ("ByEntity absent", By::ByEntity("absent".to_string())),
            ("ByTags alpha", By::ByTags(vec!["alpha".to_string()])),
            (
                "ByTags BETA gamma",
                By::ByTags(vec!["BETA".to_string(), "gamma".to_string()]),
            ),
            ("ByTags none", By::ByTags(Vec::new())),
            ("ByEpisode ep-a", By::ByEpisode(EPISODES[0].to_string())),
            ("ByEpisode ep-b", By::ByEpisode(EPISODES[1].to_string())),
            (
                "ByEpisodeSequence ep-a any",
                By::ByEpisodeSequence {
                    episode_id: EPISODES[0].to_string(),
                    min_sequence: None,
                    max_sequence: None,
                },
            ),
            (
                "ByEpisodeSequence ep-a 1-2",
                By::ByEpisodeSequence {
                    episode_id: EPISODES[0].to_string(),
                    min_sequence: Some(1),
                    max_sequence: Some(2),
                },
            ),
            (
                "ByEpisodeSequence ep-b from 2",
                By::ByEpisodeSequence {
                    episode_id: EPISODES[1].to_string(),
                    min_sequence: Some(2),
                    max_sequence: None,
                },
            ),
            (
                "ByEpisodeSequence ep-b to 0",
                By::ByEpisodeSequence {
                    episode_id: EPISODES[1].to_string(),
                    min_sequence: None,
                    max_sequence: Some(0),
                },
            ),
            ("ByRobot rover-1", By::ByRobot(ROBOTS[0].to_string())),
            ("ByMission south", By::ByMission(MISSIONS[1].to_string())),
            ("ByLocation London", near(SITES[0])),
            ("ByLocation 50 km east of London", near(SITES[1])),
            ("ByLocation 50 km north of London", near(SITES[2])),
            ("ByLocation Madrid", near(SITES[4])),
            ("ByLocation east of the antimeridian", near(SITES[5])),
            ("ByLocation west of the antimeridian", near(SITES[6])),
            ("ByLocation by the pole", near(SITES[7])),
            ("ByLocation 400 km of Paris", within(SITES[3], 400_000.0)),
            ("ByLocation London, zero radius", within(SITES[0], 0.0)),
            (
                "ByLocation 25 km of the London midpoint",
                within((51.5074, 0.2311), 25_000.0),
            ),
            ("ByLocation open ocean", near(OPEN_OCEAN)),
            ("ByLocation everywhere", within((0.0, 0.0), f64::INFINITY)),
            (
                "ByActionType grasp",
                By::ByActionType(ACTIONS[0].to_string()),
            ),
            ("ByReward -1 to -0.1", reward_bounds(0, 36)),
            ("ByReward 0 to 1", reward_bounds(40, 80)),
            ("ByReward exactly -0.3", reward_bounds(28, 28)),
            ("ByReward -0.25 to 0.25", reward_bounds(30, 50)),
            (
                "Combined Observation alpha",
                By::Combined(vec![
                    By::ByType(ExperienceType::Observation),
                    By::ByTags(vec!["alpha".to_string()]),
                ]),
            ),
            (
                "Combined low importance roots",
                By::Combined(vec![importance_bounds(0, 16), By::RootsOnly]),
            ),
            (
                "Combined ep-a rover-1",
                By::Combined(vec![
                    By::ByEpisode(EPISODES[0].to_string()),
                    By::ByRobot(ROBOTS[0].to_string()),
                ]),
            ),
            ("ByParent slot 0", By::ByParent(slot_id(0))),
            ("ByParent slot 2", By::ByParent(slot_id(2))),
            ("RootsOnly", By::RootsOnly),
        ];
        let mut contents: Vec<String> = (0..CONTENT_GROUPS)
            .flat_map(|group| (0..CONTENT_WORDS.len()).map(move |word| content(group, word)))
            .collect();
        contents.push("never stored".to_string());
        let mut external_ids: Vec<String> = (0..SLOTS)
            .flat_map(|slot| (0..EXTERNAL_VARIANTS).map(move |variant| external_id(slot, variant)))
            .collect();
        external_ids.push("ext:never".to_string());
        Self {
            searches,
            contents,
            external_ids,
            cutoffs: vec![day_start(3), day_start(8)],
            memory_prefixes: (0..SLOTS)
                .map(|slot| slot_id(slot).0.as_bytes()[..1].to_vec())
                .collect(),
        }
    }
}

fn observe_slot(store: &dyn PrimaryMemoryStore, slot: usize) -> SlotObservation {
    let id = slot_id(slot);
    SlotObservation {
        get: Answer::of(store.get(&id), |memory| memory_view(&memory)),
        get_memory: Answer::of(store.get_memory(&id), |found| {
            found.as_ref().map(memory_view)
        }),
        children: Answer::of(store.get_children(&id), sorted_ids),
        ancestors: Answer::of(store.get_ancestors(&id), |ancestors| {
            ancestors.into_iter().map(|memory| memory.id).collect()
        }),
        hierarchy: Answer::of(
            store.get_hierarchy_context(&id),
            |(ancestors, memory, children)| {
                (
                    ancestors.into_iter().map(|memory| memory.id).collect(),
                    memory.id,
                    sorted_ids(children),
                )
            },
        ),
        vector_mapping: Answer::of(store.get_vector_mapping(&id), |found| {
            found.as_ref().map(MappingView::from)
        }),
        inbound_refs: Answer::of(store.has_any_inbound_refs(&id, MERGE_USER), |cited| cited),
    }
}

fn sorted_keys(mut keys: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    keys.sort();
    keys
}

fn capture_primary(store: &dyn PrimaryMemoryStore, probes: &PrimaryProbes) -> PrimarySnapshot {
    let kv = store.key_value_store();
    let default_kv = store.default_key_value_store();
    let (all, live_stamp_bytes) = match store.get_all() {
        Ok(memories) => {
            let (records, stamp_bytes) = memory_records(&memories);
            (Answer::Value(records), stamp_bytes)
        }
        Err(_) => (Answer::Refused, 0),
    };
    PrimarySnapshot {
        all,
        all_ids: Answer::of(store.get_all_ids(), |mut ids| {
            ids.sort();
            ids
        }),
        slots: (0..SLOTS).map(|slot| observe_slot(store, slot)).collect(),
        searches: probes
            .searches
            .iter()
            .map(|(_, criteria)| Answer::of(store.search(criteria.clone()), sorted_ids))
            .collect(),
        empty_combined: Answer::of(
            store.search(SearchCriteria::Combined(Vec::new())),
            sorted_ids,
        ),
        content_hash: probes
            .contents
            .iter()
            .map(|text| store.get_by_content_hash(text))
            .collect(),
        external: probes
            .external_ids
            .iter()
            .map(|external| {
                Answer::of(store.find_by_external_id(external), |found| {
                    found.as_ref().map(memory_view)
                })
            })
            .collect(),
        uncompressed: probes
            .cutoffs
            .iter()
            .map(|cutoff| Answer::of(store.get_uncompressed_older_than(*cutoff), sorted_ids))
            .collect(),
        stats: Answer::of(store.get_stats(), |stats| StatsView {
            total_count: stats.total_count,
            compressed_count: stats.compressed_count,
            total_size_bytes_net_of_clock_stamps: stats
                .total_size_bytes
                .saturating_sub(live_stamp_bytes),
            average_importance_bits: stats.average_importance.to_bits(),
            importance_sum_bits: stats.importance_sum.to_bits(),
            total_retrievals: stats.total_retrievals,
        }),
        all_mappings: Answer::of(store.get_all_vector_mappings(), |mappings| {
            let mut views: Vec<(MemoryId, MappingView)> = mappings
                .iter()
                .map(|(id, entry)| (id.clone(), MappingView::from(entry)))
                .collect();
            views.sort_by(|a, b| a.0.cmp(&b.0));
            views
        }),
        interference: Answer::of(store.load_all_interference_records(), |(history, total)| {
            (
                history
                    .into_iter()
                    .map(|(holder, records)| (holder, json(&records)))
                    .collect(),
                total,
            )
        }),
        watermarks: WATERMARK_USERS
            .iter()
            .map(|user| store.get_fact_watermark(user))
            .collect(),
        kv: Answer::of(kv.scan_prefix(b"", None), |rows| rows),
        default_kv: DEFAULT_KV_PREFIXES
            .iter()
            .map(|prefix| Answer::of(default_kv.scan_prefix(prefix, None), |rows| rows))
            .collect(),
        memory_prefix_keys: probes
            .memory_prefixes
            .iter()
            .map(|prefix| Answer::of(store.scan_prefix_keys(prefix), sorted_keys))
            .collect(),
        foreign_prefix_keys: FOREIGN_KEY_PREFIXES
            .iter()
            .map(|prefix| Answer::of(store.scan_prefix_keys(prefix), sorted_keys))
            .collect(),
    }
}

fn compare_primary(
    comparison: &mut Comparison,
    left: &PrimarySnapshot,
    right: &PrimarySnapshot,
    probes: &PrimaryProbes,
) {
    comparison.equal("get_all", &left.all, &right.all);
    comparison.equal("get_all_ids", &left.all_ids, &right.all_ids);

    for (slot, (l, r)) in left.slots.iter().zip(&right.slots).enumerate() {
        comparison.equal(format_args!("slot {slot} get"), &l.get, &r.get);
        comparison.equal(
            format_args!("slot {slot} get_memory"),
            &l.get_memory,
            &r.get_memory,
        );
        comparison.equal(
            format_args!("slot {slot} get_children"),
            &l.children,
            &r.children,
        );
        comparison.equal(
            format_args!("slot {slot} get_ancestors"),
            &l.ancestors,
            &r.ancestors,
        );
        comparison.equal(
            format_args!("slot {slot} get_hierarchy_context"),
            &l.hierarchy,
            &r.hierarchy,
        );
        comparison.equal(
            format_args!("slot {slot} get_vector_mapping"),
            &l.vector_mapping,
            &r.vector_mapping,
        );
        comparison.equal(
            format_args!("slot {slot} has_any_inbound_refs"),
            &l.inbound_refs,
            &r.inbound_refs,
        );
    }

    for ((label, _), (l, r)) in probes
        .searches
        .iter()
        .zip(left.searches.iter().zip(&right.searches))
    {
        comparison.equal(format_args!("search {label}"), l, r);
    }
    comparison.equal(
        "search Combined([])",
        &left.empty_combined,
        &right.empty_combined,
    );

    for (text, (l, r)) in probes
        .contents
        .iter()
        .zip(left.content_hash.iter().zip(&right.content_hash))
    {
        comparison.equal(format_args!("get_by_content_hash {text:?}"), l, r);
    }
    for (external, (l, r)) in probes
        .external_ids
        .iter()
        .zip(left.external.iter().zip(&right.external))
    {
        comparison.equal(format_args!("find_by_external_id {external:?}"), l, r);
    }
    for (cutoff, (l, r)) in probes
        .cutoffs
        .iter()
        .zip(left.uncompressed.iter().zip(&right.uncompressed))
    {
        comparison.equal(format_args!("get_uncompressed_older_than {cutoff}"), l, r);
    }

    comparison.equal("get_stats", &left.stats, &right.stats);
    comparison.equal(
        "get_all_vector_mappings",
        &left.all_mappings,
        &right.all_mappings,
    );
    comparison.equal(
        "load_all_interference_records",
        &left.interference,
        &right.interference,
    );
    comparison.equal("get_fact_watermark", &left.watermarks, &right.watermarks);
    comparison.equal("key_value_store scan", &left.kv, &right.kv);
    for (prefix, (l, r)) in DEFAULT_KV_PREFIXES
        .iter()
        .zip(left.default_kv.iter().zip(&right.default_kv))
    {
        comparison.equal(
            format_args!(
                "default_key_value_store scan {:?}",
                String::from_utf8_lossy(prefix)
            ),
            l,
            r,
        );
    }
    for (prefix, (l, r)) in probes.memory_prefixes.iter().zip(
        left.memory_prefix_keys
            .iter()
            .zip(&right.memory_prefix_keys),
    ) {
        comparison.equal(format_args!("scan_prefix_keys {prefix:02x?}"), l, r);
    }
    for (prefix, (l, r)) in FOREIGN_KEY_PREFIXES.iter().zip(
        left.foreign_prefix_keys
            .iter()
            .zip(&right.foreign_prefix_keys),
    ) {
        comparison.equal(
            format_args!("scan_prefix_keys {:?}", String::from_utf8_lossy(prefix)),
            l,
            r,
        );
    }
}

fn check_primary_sequence(ops: &[PrimaryOp]) -> Result<(), TestCaseError> {
    let subjects = subjects(test_support::primary_stores, |_| None);
    let probes = PrimaryProbes::new();
    let capture = |subjects: &[Subject<Arc<dyn PrimaryMemoryStore>>]| -> Vec<PrimarySnapshot> {
        subjects
            .iter()
            .map(|subject| capture_primary(subject.store.as_ref(), &probes))
            .collect()
    };

    let compare = |c: &mut Comparison, l: &PrimarySnapshot, r: &PrimarySnapshot| {
        compare_primary(c, l, r, &probes)
    };
    agree(&subjects, &capture(&subjects), "before any step", compare)?;
    for (index, op) in ops.iter().enumerate() {
        let action = resolve_primary(op);
        let context = format!("step {index} ({})", action.describe());
        let outcomes: Vec<Outcome> = subjects
            .iter()
            .map(|subject| apply_primary(subject.store.as_ref(), &action))
            .collect();
        agree_on_outcomes(&subjects, &outcomes, &context)?;
        agree(&subjects, &capture(&subjects), &context, compare)?;
    }
    Ok(())
}

// =============================================================================
// KeyValueStore
// =============================================================================

const KV_FAMILY: &str = "differential_kv";
const KV_HARNESS_KEYS: [&[u8]; 10] = [
    b"",
    b"a",
    b"ab",
    b"abc",
    b"b",
    b"\x00",
    b"\x00\x01",
    b"\xff",
    b"\xff\xff",
    b"a\xff",
];
const KV_SCAN_PREFIXES: [&[u8]; 6] = [b"", b"a", b"ab", b"\x00", b"\xff", b"z"];

fn kv_ops() -> impl Strategy<Value = Vec<KvOp>> {
    prop::collection::vec(kv_op(KV_HARNESS_KEYS.len()), 1..=MAX_OPS)
}

#[derive(Debug, PartialEq)]
struct KvSnapshot {
    gets: Vec<Answer<Option<Vec<u8>>>>,
    scans: Vec<Answer<Rows>>,
}

fn capture_kv(store: &dyn KeyValueStore) -> KvSnapshot {
    KvSnapshot {
        gets: KV_HARNESS_KEYS
            .iter()
            .map(|key| Answer::of(store.get_bytes(key), |value| value))
            .collect(),
        scans: KV_SCAN_PREFIXES
            .iter()
            .flat_map(|prefix| {
                SCAN_LIMITS
                    .iter()
                    .map(move |limit| Answer::of(store.scan_prefix(prefix, *limit), |rows| rows))
            })
            .collect(),
    }
}

fn compare_kv(comparison: &mut Comparison, left: &KvSnapshot, right: &KvSnapshot) {
    for (key, (l, r)) in KV_HARNESS_KEYS
        .iter()
        .zip(left.gets.iter().zip(&right.gets))
    {
        comparison.equal(format_args!("get_bytes {key:02x?}"), l, r);
    }
    let probes = KV_SCAN_PREFIXES
        .iter()
        .flat_map(|prefix| SCAN_LIMITS.iter().map(move |limit| (prefix, limit)));
    for ((prefix, limit), (l, r)) in probes.zip(left.scans.iter().zip(&right.scans)) {
        comparison.equal(
            format_args!("scan_prefix {prefix:02x?} limit {limit:?}"),
            l,
            r,
        );
    }
}

fn check_kv_sequence(ops: &[KvOp]) -> Result<(), TestCaseError> {
    let subjects = subjects(
        || test_support::key_value_store_implementations(KV_FAMILY),
        |named: &NamedStore<dyn KeyValueStore>| Some(named.implementation),
    );
    let capture = |subjects: &[Subject<NamedStore<dyn KeyValueStore>>]| -> Vec<KvSnapshot> {
        subjects
            .iter()
            .map(|subject| capture_kv(subject.store.store.as_ref()))
            .collect()
    };

    agree(
        &subjects,
        &capture(&subjects),
        "before any step",
        compare_kv,
    )?;
    for (index, op) in ops.iter().enumerate() {
        let action = KvAction::resolve(op, &KV_HARNESS_KEYS);
        let context = format!("step {index} ({action:?})");
        let outcomes: Vec<Outcome> = subjects
            .iter()
            .map(|subject| apply_kv(subject.store.store.as_ref(), &action))
            .collect();
        agree_on_outcomes(&subjects, &outcomes, &context)?;
        agree(&subjects, &capture(&subjects), &context, compare_kv)?;
    }
    Ok(())
}

// =============================================================================
// AuditStore
// =============================================================================

/// `ali` is a byte prefix of `alice` and `ops` of `ops:west`, the two ways a
/// `{user}:` key prefix can be mistaken for another user's.
const AUDIT_USERS: [&str; 4] = ["alice", "ali", "ops:west", "ops"];
const AUDIT_EVENT_TYPES: [&str; 3] = ["write", "forget", "recall"];
const AUDIT_LIST_LIMITS: [usize; 4] = [usize::MAX, 0, 1, 2];

#[derive(Debug, Clone)]
struct AuditSpec {
    second: i64,
    event_type: usize,
    memory: usize,
    details: usize,
}

prop_compose! {
    fn audit_spec()(
        second in 0i64..40,
        event_type in 0..AUDIT_EVENT_TYPES.len(),
        memory in 0..SLOTS,
        details in 0..CONTENT_WORDS.len(),
    ) -> AuditSpec {
        AuditSpec { second, event_type, memory, details }
    }
}

#[derive(Debug, Clone)]
enum AuditOp {
    Append {
        user: usize,
        spec: AuditSpec,
    },
    Rotate {
        user: usize,
        max_entries: usize,
        retention_second: i64,
    },
    Flush,
}

fn audit_ops() -> impl Strategy<Value = Vec<AuditOp>> {
    let op = prop_oneof![
        8 => (0..AUDIT_USERS.len(), audit_spec())
            .prop_map(|(user, spec)| AuditOp::Append { user, spec }),
        2 => (0..AUDIT_USERS.len(), 0usize..4, 0i64..40).prop_map(
            |(user, max_entries, retention_second)| AuditOp::Rotate { user, max_entries, retention_second },
        ),
        1 => Just(AuditOp::Flush),
    ];
    prop::collection::vec(op, 1..=MAX_OPS)
}

fn apply_audit(store: &dyn AuditStore, op: &AuditOp) -> Outcome {
    match op {
        AuditOp::Append { user, spec } => {
            let entry = AuditLogEntry {
                timestamp: epoch() + Duration::seconds(spec.second),
                event_type: AUDIT_EVENT_TYPES[spec.event_type].to_string(),
                memory_id: slot_id(spec.memory).0.to_string(),
                details: CONTENT_WORDS[spec.details].to_string(),
            };
            done(store.append_event(AUDIT_USERS[*user], &entry))
        }
        AuditOp::Rotate {
            user,
            max_entries,
            retention_second,
        } => counted(store.rotate_events(
            AUDIT_USERS[*user],
            *max_entries,
            epoch() + Duration::seconds(*retention_second),
        )),
        AuditOp::Flush => done(store.flush()),
    }
}

#[derive(Debug, PartialEq)]
struct AuditSnapshot {
    user_ids: Answer<Vec<String>>,
    events: Vec<Answer<Vec<AuditLogEntry>>>,
}

fn capture_audit(store: &dyn AuditStore) -> AuditSnapshot {
    AuditSnapshot {
        user_ids: Answer::of(store.list_user_ids(), |mut ids| {
            ids.sort();
            ids
        }),
        events: AUDIT_USERS
            .iter()
            .flat_map(|user| {
                AUDIT_LIST_LIMITS
                    .iter()
                    .map(move |limit| Answer::of(store.list_events(user, *limit), |events| events))
            })
            .collect(),
    }
}

fn compare_audit(comparison: &mut Comparison, left: &AuditSnapshot, right: &AuditSnapshot) {
    comparison.equal("list_user_ids", &left.user_ids, &right.user_ids);
    let probes = AUDIT_USERS
        .iter()
        .flat_map(|user| AUDIT_LIST_LIMITS.iter().map(move |limit| (user, limit)));
    for ((user, limit), (l, r)) in probes.zip(left.events.iter().zip(&right.events)) {
        comparison.equal(format_args!("list_events {user:?} limit {limit}"), l, r);
    }
}

fn check_audit_sequence(ops: &[AuditOp]) -> Result<(), TestCaseError> {
    let subjects = subjects(test_support::audit_stores, |_| None);
    let capture = |subjects: &[Subject<Arc<dyn AuditStore>>]| -> Vec<AuditSnapshot> {
        subjects
            .iter()
            .map(|subject| capture_audit(subject.store.as_ref()))
            .collect()
    };

    agree(
        &subjects,
        &capture(&subjects),
        "before any step",
        compare_audit,
    )?;
    for (index, op) in ops.iter().enumerate() {
        let context = format!("step {index} ({op:?})");
        let outcomes: Vec<Outcome> = subjects
            .iter()
            .map(|subject| apply_audit(subject.store.as_ref(), op))
            .collect();
        agree_on_outcomes(&subjects, &outcomes, &context)?;
        agree(&subjects, &capture(&subjects), &context, compare_audit)?;
    }
    Ok(())
}

// =============================================================================
// BatchWriter, read back through ColumnStore
// =============================================================================

const DECLARED_FAMILIES: [&str; 3] = ["alpha", "beta", "gamma"];
const READ_FAMILIES: [&str; 4] = ["default", "alpha", "beta", "gamma"];
const BATCH_KEYS: [&[u8]; 6] = [b"k", b"k1", b"k2", b"pre:a", b"pre:b", b"\xff"];
const BATCH_PREFIXES: [Option<&[u8]>; 4] = [None, Some(b""), Some(b"k"), Some(b"pre:")];

#[derive(Debug, Clone, Copy)]
enum BatchFamily {
    Unnamed,
    Named(usize),
    Undeclared,
}

impl BatchFamily {
    fn name(self) -> Option<&'static str> {
        match self {
            BatchFamily::Unnamed => None,
            BatchFamily::Named(index) => Some(READ_FAMILIES[index % READ_FAMILIES.len()]),
            BatchFamily::Undeclared => Some(UNDECLARED_FAMILY),
        }
    }
}

#[derive(Debug, Clone)]
struct BatchEntry {
    family: BatchFamily,
    key: usize,
    value: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
enum BatchHarnessOp {
    Batch(Vec<BatchEntry>),
    ColumnPut {
        family: usize,
        key: usize,
        value: Vec<u8>,
    },
    ColumnDelete {
        family: usize,
        key: usize,
    },
    Flush {
        family: usize,
    },
}

fn batch_entry() -> impl Strategy<Value = BatchEntry> {
    let family = prop_oneof![
        2 => Just(BatchFamily::Unnamed),
        6 => (0..READ_FAMILIES.len()).prop_map(BatchFamily::Named),
        1 => Just(BatchFamily::Undeclared),
    ];
    (family, 0..BATCH_KEYS.len(), prop::option::of(value_bytes()))
        .prop_map(|(family, key, value)| BatchEntry { family, key, value })
}

fn batch_ops() -> impl Strategy<Value = Vec<BatchHarnessOp>> {
    let op = prop_oneof![
        6 => prop::collection::vec(batch_entry(), 0..6).prop_map(BatchHarnessOp::Batch),
        2 => (0..READ_FAMILIES.len(), 0..BATCH_KEYS.len(), value_bytes())
            .prop_map(|(family, key, value)| BatchHarnessOp::ColumnPut { family, key, value }),
        1 => (0..READ_FAMILIES.len(), 0..BATCH_KEYS.len())
            .prop_map(|(family, key)| BatchHarnessOp::ColumnDelete { family, key }),
        1 => (0..READ_FAMILIES.len()).prop_map(|family| BatchHarnessOp::Flush { family }),
    ];
    prop::collection::vec(op, 1..=MAX_OPS)
}

fn build_batch(entries: &[BatchEntry]) -> WriteBatch {
    let mut batch = WriteBatch::new();
    for entry in entries {
        let key = BATCH_KEYS[entry.key].to_vec();
        match (entry.family.name(), &entry.value) {
            (None, Some(value)) => batch.put(key, value.clone()),
            (None, None) => batch.delete(key),
            (Some(family), Some(value)) => batch.put_cf(family, key, value.clone()),
            (Some(family), None) => batch.delete_cf(family, key),
        }
    }
    batch
}

fn apply_batch_op(subject: &BatchWriterUnderTest, op: &BatchHarnessOp) -> Outcome {
    match op {
        BatchHarnessOp::Batch(entries) => done(subject.writer.write_batch(&build_batch(entries))),
        BatchHarnessOp::ColumnPut { family, key, value } => done(subject.columns.put_cf(
            READ_FAMILIES[*family],
            BATCH_KEYS[*key],
            value,
        )),
        BatchHarnessOp::ColumnDelete { family, key } => done(
            subject
                .columns
                .delete_cf(READ_FAMILIES[*family], BATCH_KEYS[*key]),
        ),
        BatchHarnessOp::Flush { family } => done(subject.columns.flush_cf(READ_FAMILIES[*family])),
    }
}

#[derive(Debug, PartialEq)]
struct ColumnSnapshot {
    scans: Vec<Answer<Rows>>,
    gets: Vec<Answer<Option<Vec<u8>>>>,
    undeclared_get: Answer<Option<Vec<u8>>>,
    undeclared_scan: Answer<Rows>,
}

fn column_scan_probes() -> impl Iterator<Item = (&'static str, Option<&'static [u8]>, Option<usize>)>
{
    READ_FAMILIES.iter().flat_map(|family| {
        BATCH_PREFIXES.iter().flat_map(move |prefix| {
            SCAN_LIMITS
                .iter()
                .map(move |limit| (*family, *prefix, *limit))
        })
    })
}

fn column_get_probes() -> impl Iterator<Item = (&'static str, &'static [u8])> {
    READ_FAMILIES
        .iter()
        .flat_map(|family| BATCH_KEYS.iter().map(move |key| (*family, *key)))
}

fn capture_columns(columns: &dyn ColumnStore) -> ColumnSnapshot {
    ColumnSnapshot {
        scans: column_scan_probes()
            .map(|(family, prefix, limit)| {
                Answer::of(columns.scan_cf(family, prefix, limit), |rows| rows)
            })
            .collect(),
        gets: column_get_probes()
            .map(|(family, key)| Answer::of(columns.get_cf(family, key), |value| value))
            .collect(),
        undeclared_get: Answer::of(columns.get_cf(UNDECLARED_FAMILY, BATCH_KEYS[0]), |value| {
            value
        }),
        undeclared_scan: Answer::of(columns.scan_cf(UNDECLARED_FAMILY, None, None), |rows| rows),
    }
}

fn compare_columns(comparison: &mut Comparison, left: &ColumnSnapshot, right: &ColumnSnapshot) {
    for ((family, prefix, limit), (l, r)) in
        column_scan_probes().zip(left.scans.iter().zip(&right.scans))
    {
        comparison.equal(
            format_args!("scan_cf {family} prefix {prefix:02x?} limit {limit:?}"),
            l,
            r,
        );
    }
    for ((family, key), (l, r)) in column_get_probes().zip(left.gets.iter().zip(&right.gets)) {
        comparison.equal(format_args!("get_cf {family} {key:02x?}"), l, r);
    }
    comparison.equal(
        "get_cf of an undeclared family",
        &left.undeclared_get,
        &right.undeclared_get,
    );
    comparison.equal(
        "scan_cf of an undeclared family",
        &left.undeclared_scan,
        &right.undeclared_scan,
    );
}

fn check_batch_sequence(ops: &[BatchHarnessOp]) -> Result<(), TestCaseError> {
    let subjects = subjects(|| test_support::batch_writers(&DECLARED_FAMILIES), |_| None);
    let capture = |subjects: &[Subject<BatchWriterUnderTest>]| -> Vec<ColumnSnapshot> {
        subjects
            .iter()
            .map(|subject| capture_columns(subject.store.columns.as_ref()))
            .collect()
    };

    agree(
        &subjects,
        &capture(&subjects),
        "before any step",
        compare_columns,
    )?;
    for (index, op) in ops.iter().enumerate() {
        let context = format!("step {index} ({op:?})");
        let outcomes: Vec<Outcome> = subjects
            .iter()
            .map(|subject| apply_batch_op(&subject.store, op))
            .collect();
        agree_on_outcomes(&subjects, &outcomes, &context)?;
        agree(&subjects, &capture(&subjects), &context, compare_columns)?;
    }
    Ok(())
}

// =============================================================================
// GraphStore
// =============================================================================

const GRAPH_SLOTS: usize = 6;
const ENTITY_NAMES: [&str; GRAPH_SLOTS] =
    ["harbor", "lantern", "meadow", "quartz", "summit", "willow"];
const ENTITY_SUMMARIES: [&str; 3] = ["", "seen near the dock", "a recurring landmark"];
const ENTITY_ROLES: [&str; 2] = ["anchor", "waypoint"];
const EDGE_CONTEXTS: [&str; 3] = ["", "co-mentioned", "named together in one sentence"];
const EPISODE_WORDS: [&str; 3] = ["dawn patrol", "harbour survey", "night watch"];

fn graph_uuid(kind: u8, slot: usize) -> Uuid {
    let mut bytes = [0u8; 16];
    bytes[0] = kind;
    bytes[1] = slot as u8;
    bytes[15] = 0x6A;
    Uuid::from_bytes(bytes)
}

fn entity_uuid(slot: usize) -> Uuid {
    graph_uuid(0xE0, slot)
}

fn edge_uuid(slot: usize) -> Uuid {
    graph_uuid(0xD0, slot)
}

fn episode_uuid(slot: usize) -> Uuid {
    graph_uuid(0xC0, slot)
}

const RELATION_TYPE_COUNT: usize = 3;

fn relation_type(index: usize) -> RelationType {
    match index % RELATION_TYPE_COUNT {
        0 => RelationType::WorksWith,
        1 => RelationType::PartOf,
        _ => RelationType::RelatedTo,
    }
}

fn entity_label(index: usize) -> EntityLabel {
    match index {
        0 => EntityLabel::Person,
        1 => EntityLabel::Technology,
        2 => EntityLabel::Concept,
        3 => EntityLabel::Location,
        _ => EntityLabel::Other("landmark".to_string()),
    }
}

#[derive(Debug, Clone)]
struct EntitySpec {
    labels: Vec<usize>,
    summary: usize,
    role: Option<usize>,
    salience_tenths: u8,
    proper_noun: bool,
    seen_after_hours: i64,
}

/// Endpoints and type are drawn per write, so a rewrite under one uuid can move an
/// edge between entities, and two uuids can share a pair and a type.
#[derive(Debug, Clone)]
struct EdgeSpec {
    from: usize,
    to: usize,
    relation: usize,
    strength_tenths: u8,
    context: usize,
    activation_count: u32,
    source_episode: Option<usize>,
    confidence_tenths: Option<u8>,
}

#[derive(Debug, Clone)]
struct EpisodeSpec {
    word: usize,
    entity_refs: Vec<usize>,
    source: usize,
    tagged: bool,
    day: i64,
}

#[derive(Debug, Clone)]
enum GraphOp {
    PutEntity { slot: usize, spec: EntitySpec },
    PutRelationship { slot: usize, spec: EdgeSpec },
    PutEpisode { slot: usize, spec: EpisodeSpec },
    Flush,
}

prop_compose! {
    fn entity_spec()(
        labels in prop::collection::btree_set(0usize..5, 0..3),
        summary in 0..ENTITY_SUMMARIES.len(),
        role in prop::option::of(0..ENTITY_ROLES.len()),
        salience_tenths in 0u8..=10,
        proper_noun in any::<bool>(),
        seen_after_hours in 0i64..48,
    ) -> EntitySpec {
        EntitySpec {
            labels: labels.into_iter().collect(),
            summary,
            role,
            salience_tenths,
            proper_noun,
            seen_after_hours,
        }
    }
}

prop_compose! {
    fn edge_spec()(
        from in 0..GRAPH_SLOTS,
        to in 0..GRAPH_SLOTS,
        relation in 0..RELATION_TYPE_COUNT,
        strength_tenths in 0u8..=10,
        context in 0..EDGE_CONTEXTS.len(),
        activation_count in 0u32..3,
        source_episode in prop::option::of(0..GRAPH_SLOTS),
        confidence_tenths in prop::option::of(0u8..=10),
    ) -> EdgeSpec {
        EdgeSpec {
            from,
            to,
            relation,
            strength_tenths,
            context,
            activation_count,
            source_episode,
            confidence_tenths,
        }
    }
}

prop_compose! {
    fn episode_spec()(
        word in 0..EPISODE_WORDS.len(),
        entity_refs in prop::collection::btree_set(0..GRAPH_SLOTS, 0..3),
        source in 0usize..4,
        tagged in any::<bool>(),
        day in 0i64..7,
    ) -> EpisodeSpec {
        EpisodeSpec { word, entity_refs: entity_refs.into_iter().collect(), source, tagged, day }
    }
}

fn graph_ops() -> impl Strategy<Value = Vec<GraphOp>> {
    let op = prop_oneof![
        4 => (0..GRAPH_SLOTS, entity_spec())
            .prop_map(|(slot, spec)| GraphOp::PutEntity { slot, spec }),
        4 => (0..GRAPH_SLOTS, edge_spec())
            .prop_map(|(slot, spec)| GraphOp::PutRelationship { slot, spec }),
        3 => (0..GRAPH_SLOTS, episode_spec())
            .prop_map(|(slot, spec)| GraphOp::PutEpisode { slot, spec }),
        1 => Just(GraphOp::Flush),
    ];
    prop::collection::vec(op, 1..=MAX_OPS)
}

fn build_entity(slot: usize, spec: &EntitySpec) -> EntityNode {
    let labels: Vec<EntityLabel> = spec
        .labels
        .iter()
        .map(|index| entity_label(*index))
        .collect();
    EntityNode {
        uuid: entity_uuid(slot),
        name: ENTITY_NAMES[slot].to_string(),
        pii_classification: PiiClassification::from_labels(&labels),
        labels,
        created_at: epoch(),
        last_seen_at: epoch() + Duration::hours(spec.seen_after_hours),
        mention_count: 1,
        summary: ENTITY_SUMMARIES[spec.summary].to_string(),
        attributes: spec
            .role
            .map(|role| HashMap::from([("role".to_string(), ENTITY_ROLES[role].to_string())]))
            .unwrap_or_default(),
        name_embedding: None,
        salience: f32::from(spec.salience_tenths) / 10.0,
        is_proper_noun: spec.proper_noun,
    }
}

fn build_edge(slot: usize, spec: &EdgeSpec) -> RelationshipEdge {
    RelationshipEdge {
        uuid: edge_uuid(slot),
        from_entity: entity_uuid(spec.from),
        to_entity: entity_uuid(spec.to),
        relation_type: relation_type(spec.relation),
        strength: f32::from(spec.strength_tenths) / 10.0,
        created_at: epoch(),
        valid_at: epoch() + Duration::hours(1),
        invalidated_at: None,
        source_episode_id: spec.source_episode.map(episode_uuid),
        context: EDGE_CONTEXTS[spec.context].to_string(),
        last_activated: epoch() + Duration::hours(2),
        activation_count: spec.activation_count,
        ltp_status: Default::default(),
        tier: Default::default(),
        activation_timestamps: None,
        entity_confidence: spec
            .confidence_tenths
            .map(|tenths| f32::from(tenths) / 10.0),
        created_by: Default::default(),
        forward_strength: 0.0,
        backward_strength: 0.0,
        strengthening_sources: Default::default(),
        last_corroborated: None,
    }
}

fn build_episode(slot: usize, spec: &EpisodeSpec) -> EpisodicNode {
    EpisodicNode {
        uuid: episode_uuid(slot),
        name: format!("episode {slot}"),
        content: EPISODE_WORDS[spec.word].to_string(),
        valid_at: day_start(spec.day),
        created_at: epoch(),
        entity_refs: spec
            .entity_refs
            .iter()
            .map(|other| entity_uuid(*other))
            .collect(),
        source: match spec.source {
            0 => EpisodeSource::Message,
            1 => EpisodeSource::Document,
            2 => EpisodeSource::Event,
            _ => EpisodeSource::Observation,
        },
        metadata: if spec.tagged {
            HashMap::from([("channel".to_string(), "radio".to_string())])
        } else {
            HashMap::new()
        },
    }
}

enum GraphAction {
    PutEntity(Box<EntityNode>),
    PutRelationship(Box<RelationshipEdge>),
    PutEpisode(Box<EpisodicNode>),
    Flush,
}

fn resolve_graph(op: &GraphOp) -> GraphAction {
    match op {
        GraphOp::PutEntity { slot, spec } => {
            GraphAction::PutEntity(Box::new(build_entity(*slot, spec)))
        }
        GraphOp::PutRelationship { slot, spec } => {
            GraphAction::PutRelationship(Box::new(build_edge(*slot, spec)))
        }
        GraphOp::PutEpisode { slot, spec } => {
            GraphAction::PutEpisode(Box::new(build_episode(*slot, spec)))
        }
        GraphOp::Flush => GraphAction::Flush,
    }
}

fn apply_graph(store: &dyn GraphStore, action: &GraphAction) -> Outcome {
    match action {
        GraphAction::PutEntity(entity) => done(store.put_entity(entity)),
        GraphAction::PutRelationship(edge) => done(store.put_relationship(edge)),
        GraphAction::PutEpisode(episode) => done(store.put_episode(episode)),
        GraphAction::Flush => done(store.flush()),
    }
}

/// The edges listed for an entity, in uuid order: the listing order is the engine's.
fn edge_views(mut edges: Vec<RelationshipEdge>) -> Vec<Value> {
    edges.sort_by_key(|edge| edge.uuid);
    edges.iter().map(json).collect()
}

/// What one slot's uuids read back as: the entity, edge and episode written under
/// that slot, and the edges listed for the slot's entity.
#[derive(Debug, PartialEq)]
struct GraphSlotObservation {
    entity: Answer<Option<Value>>,
    relationship: Answer<Option<Value>>,
    episode: Answer<Option<Value>>,
    edges: Answer<Vec<Value>>,
}

#[derive(Debug, PartialEq)]
struct GraphSnapshot {
    stats: Answer<(usize, usize, usize)>,
    slots: Vec<GraphSlotObservation>,
}

fn capture_graph(store: &dyn GraphStore) -> GraphSnapshot {
    GraphSnapshot {
        stats: Answer::of(store.stats(), |stats| {
            (
                stats.entity_count,
                stats.relationship_count,
                stats.episode_count,
            )
        }),
        slots: (0..GRAPH_SLOTS)
            .map(|slot| GraphSlotObservation {
                entity: Answer::of(store.get_entity(&entity_uuid(slot)), |found| {
                    found.as_ref().map(json)
                }),
                relationship: Answer::of(store.get_relationship(&edge_uuid(slot)), |found| {
                    found.as_ref().map(json)
                }),
                episode: Answer::of(store.get_episode(&episode_uuid(slot)), |found| {
                    found.as_ref().map(json)
                }),
                edges: Answer::of(
                    store.list_relationships_for_entity(&entity_uuid(slot)),
                    edge_views,
                ),
            })
            .collect(),
    }
}

fn compare_graph(comparison: &mut Comparison, left: &GraphSnapshot, right: &GraphSnapshot) {
    comparison.equal("stats", &left.stats, &right.stats);
    for (slot, (l, r)) in left.slots.iter().zip(&right.slots).enumerate() {
        comparison.equal(format_args!("get_entity slot {slot}"), &l.entity, &r.entity);
        comparison.equal(
            format_args!("get_relationship slot {slot}"),
            &l.relationship,
            &r.relationship,
        );
        comparison.equal(
            format_args!("get_episode slot {slot}"),
            &l.episode,
            &r.episode,
        );
        comparison.equal(
            format_args!("list_relationships_for_entity slot {slot}"),
            &l.edges,
            &r.edges,
        );
    }
}

fn check_graph_sequence(ops: &[GraphOp]) -> Result<(), TestCaseError> {
    let subjects = subjects(test_support::graph_stores, |_| None);
    let capture = |subjects: &[Subject<Arc<dyn GraphStore>>]| -> Vec<GraphSnapshot> {
        subjects
            .iter()
            .map(|subject| capture_graph(subject.store.as_ref()))
            .collect()
    };

    agree(
        &subjects,
        &capture(&subjects),
        "before any step",
        compare_graph,
    )?;
    for (index, op) in ops.iter().enumerate() {
        let action = resolve_graph(op);
        let context = format!("step {index} ({op:?})");
        let outcomes: Vec<Outcome> = subjects
            .iter()
            .map(|subject| apply_graph(subject.store.as_ref(), &action))
            .collect();
        agree_on_outcomes(&subjects, &outcomes, &context)?;
        agree(&subjects, &capture(&subjects), &context, compare_graph)?;
    }
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

/// One differential test per store kind, under the name that says what it proves in
/// this build: engines agreeing with each other, or one engine agreeing with its replay.
macro_rules! differential_test {
    ($engines:ident, $replay:ident, $ops:expr, $check:path) => {
        proptest! {
            #![proptest_config(harness_config())]

            #[cfg(all(feature = "rocksdb", feature = "storage-redb"))]
            #[test]
            fn $engines(ops in $ops) {
                $check(&ops)?;
            }

            #[cfg(not(all(feature = "rocksdb", feature = "storage-redb")))]
            #[test]
            fn $replay(ops in $ops) {
                $check(&ops)?;
            }
        }
    };
}

differential_test!(
    primary_memory_store_engines_agree_after_every_step,
    primary_memory_store_replay_agrees_after_every_step,
    primary_ops(),
    check_primary_sequence
);

differential_test!(
    key_value_store_engines_agree_after_every_step,
    key_value_store_replay_agrees_after_every_step,
    kv_ops(),
    check_kv_sequence
);

differential_test!(
    audit_store_engines_agree_after_every_step,
    audit_store_replay_agrees_after_every_step,
    audit_ops(),
    check_audit_sequence
);

differential_test!(
    batch_writer_engines_agree_after_every_step,
    batch_writer_replay_agrees_after_every_step,
    batch_ops(),
    check_batch_sequence
);

differential_test!(
    graph_store_engines_agree_after_every_step,
    graph_store_replay_agrees_after_every_step,
    graph_ops(),
    check_graph_sequence
);

#[test]
fn every_known_divergence_is_named_once_and_cites_both_engines() {
    let mut names = BTreeSet::new();
    for divergence in KNOWN_DIVERGENCES {
        assert!(
            names.insert(divergence.name),
            "`{}` is listed twice",
            divergence.name
        );
        assert!(
            divergence.rocksdb.starts_with("src/") && divergence.redb.starts_with("src/"),
            "`{}` must cite the source of both engines' behaviour",
            divergence.name
        );
        assert!(
            !divergence.why.is_empty(),
            "`{}` must say why it is admitted",
            divergence.name
        );
    }
}
