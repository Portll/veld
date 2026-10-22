//! Synthetic graphs for the gap pipeline, and tests that an incrementally
//! synced mirror matches a full rebuild of the same graph.

use chrono::Utc;
use parking_lot::RwLock;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use uuid::Uuid;

use crate::graph_memory::{
    EdgeSource, EdgeTier, EntityLabel, EntityNode, GraphMemory, LtpStatus, RelationType,
    RelationshipEdge,
};
use crate::memory::gap_topology::{GapDetectionConfig, GapDetector, GapScope, GapStore};
use crate::memory::graph_mirror::{sync_mirror, MirrorSyncOptions};
use crate::memory::slow_store::{SlowStore, SyncMode};

/// xorshift64, so a graph is the same on every run.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    pub(crate) fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

const LABELS: [EntityLabel; 5] = [
    EntityLabel::Person,
    EntityLabel::Concept,
    EntityLabel::Technology,
    EntityLabel::Project,
    EntityLabel::FilePath,
];

pub(crate) fn entity(name: &str, label: EntityLabel, salience: f32) -> EntityNode {
    let now = Utc::now();
    EntityNode {
        uuid: Uuid::new_v4(),
        name: name.to_string(),
        labels: vec![label],
        created_at: now,
        last_seen_at: now,
        mention_count: 1,
        summary: String::new(),
        attributes: HashMap::new(),
        name_embedding: None,
        salience,
        is_proper_noun: true,
        pii_classification: Default::default(),
    }
}

pub(crate) fn edge(
    from: Uuid,
    to: Uuid,
    relation: RelationType,
    strength: f32,
) -> RelationshipEdge {
    let now = Utc::now();
    RelationshipEdge {
        uuid: Uuid::new_v4(),
        from_entity: from,
        to_entity: to,
        relation_type: relation,
        strength,
        created_at: now,
        valid_at: now,
        invalidated_at: None,
        source_episode_id: None,
        context: String::new(),
        last_activated: now,
        activation_count: 1,
        ltp_status: LtpStatus::None,
        tier: EdgeTier::L1Working,
        activation_timestamps: None,
        entity_confidence: None,
        created_by: EdgeSource::CoOccurrence,
        forward_strength: 0.0,
        backward_strength: 0.0,
        strengthening_sources: Default::default(),
        last_corroborated: None,
    }
}

/// A graph of `entities` entities and about `edges` edges: sources skewed
/// toward low indices so hubs form, one edge in twenty a second relation on
/// a pair already linked, a sixteenth of entities in the `file:` namespace,
/// half with a small name embedding, and one `project:demo` tag entity.
pub(crate) fn synthetic_graph(
    graph: &GraphMemory,
    entities: usize,
    edges: usize,
    seed: u64,
) -> Vec<Uuid> {
    let mut rng = Rng::new(seed);
    let mut ids = Vec::with_capacity(entities + 1);
    for i in 0..entities {
        let name = if i % 16 == 0 {
            format!("file:src/m{i}.rs")
        } else {
            format!("entity-{i}")
        };
        let mut e = entity(&name, LABELS[rng.below(LABELS.len())].clone(), rng.unit());
        if i % 2 == 0 {
            e.name_embedding = Some((0..16).map(|_| rng.unit() - 0.5).collect());
        }
        e.mention_count = 1 + rng.below(50);
        graph.put_entity_record(&e).unwrap();
        ids.push(e.uuid);
    }
    let tag = entity("project:demo", EntityLabel::Project, 0.9);
    graph.put_entity_record(&tag).unwrap();
    let mut last: Option<(Uuid, Uuid)> = None;
    for i in 0..edges {
        let (from, to) = match last.filter(|_| i % 20 == 0) {
            Some(pair) => pair,
            None => {
                let skew = rng.unit() * rng.unit();
                let from = ids[(skew * entities as f32) as usize % entities];
                let mut to = ids[rng.below(entities)];
                if to == from {
                    to = ids[(rng.below(entities) + 1) % entities];
                }
                (from, to)
            }
        };
        let relation = if i % 20 == 0 {
            RelationType::PartOf
        } else {
            RelationType::RelatedTo
        };
        let e = edge(from, to, relation, 0.05 + 0.95 * rng.unit());
        graph.put_relationship_record(&e).unwrap();
        last = Some((from, to));
        if i % 97 == 0 {
            graph
                .put_relationship_record(&edge(tag.uuid, from, RelationType::RelatedTo, 0.5))
                .unwrap();
        }
    }
    ids
}

/// Entity and edge rows of a mirror, comparable across stores.
type MirrorRows = (
    BTreeSet<(String, String, u32, i64)>,
    BTreeSet<(String, String, String, String, u32)>,
);

pub(crate) fn mirror_rows(store: &SlowStore) -> MirrorRows {
    let conn = store.read().unwrap();
    let entities = conn
        .prepare("SELECT uuid, name, salience, mention_count FROM entities")
        .unwrap()
        .query_map([], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get::<_, f32>(2)?.to_bits(),
                r.get(3)?,
            ))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let edges = conn
        .prepare("SELECT uuid, from_entity, to_entity, relation_type, strength FROM edges")
        .unwrap()
        .query_map([], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get::<_, f32>(4)?.to_bits(),
            ))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    (entities, edges)
}

/// The mirror a full rebuild of `graph` produces now.
fn rebuilt(graph: &GraphMemory, dir: &std::path::Path) -> SlowStore {
    let store = SlowStore::open(&dir.join("rebuilt.db")).unwrap();
    store
        .sync_from_graph(
            &graph.get_all_entities().unwrap(),
            &graph.get_all_relationships().unwrap(),
        )
        .unwrap();
    store
}

pub(crate) fn options(chunk_rows: usize) -> MirrorSyncOptions {
    MirrorSyncOptions {
        ttl: std::time::Duration::ZERO,
        chunk_rows,
    }
}

/// Every kind of graph write the mirror must follow, against `ids`.
fn mutate(graph: &GraphMemory, ids: &[Uuid], rng: &mut Rng) {
    let edges = graph.get_all_relationships().unwrap();
    for (i, e) in edges.iter().take(40).enumerate() {
        let mut e = e.clone();
        match i % 5 {
            0 => e.strength = (e.strength + 0.3).min(1.0),
            1 => e.strength *= 0.2,
            2 => {
                graph.delete_relationship(&e.uuid).unwrap();
                continue;
            }
            3 => {
                graph.invalidate_relationship(&e.uuid).unwrap();
                continue;
            }
            _ => e.to_entity = ids[rng.below(ids.len())],
        }
        graph.put_relationship_record(&e).unwrap();
    }
    // A stronger second relation on a linked pair.
    if let Some(e) = edges.get(50) {
        graph
            .put_relationship_record(&edge(
                e.from_entity,
                e.to_entity,
                RelationType::WorksWith,
                1.0,
            ))
            .unwrap();
    }
    for uuid in ids.iter().skip(7).step_by(41).take(5) {
        graph.delete_entity(uuid).unwrap();
    }
    for i in 0..30 {
        let e = entity(&format!("late-{i}"), EntityLabel::Concept, rng.unit());
        graph.put_entity_record(&e).unwrap();
        let other = ids[rng.below(ids.len())];
        graph
            .put_relationship_record(&edge(e.uuid, other, RelationType::RelatedTo, rng.unit()))
            .unwrap();
    }
}

fn open_graph(dir: &std::path::Path) -> Arc<RwLock<GraphMemory>> {
    Arc::new(RwLock::new(GraphMemory::new(dir, None).unwrap()))
}

#[test]
fn an_incremental_sync_leaves_the_mirror_a_full_rebuild_would_build() {
    let dir = tempfile::tempdir().unwrap();
    let graph = open_graph(&dir.path().join("graph"));
    let ids = synthetic_graph(&graph.read(), 300, 2_000, 7);
    let store = SlowStore::open(&dir.path().join("mirror.db")).unwrap();

    let first = sync_mirror(&graph, &store, &options(64)).unwrap();
    assert_eq!(first.mode, SyncMode::Full);
    assert_eq!(first.changed_during_copy, Some(0));
    assert_eq!(
        mirror_rows(&store),
        mirror_rows(&rebuilt(&graph.read(), dir.path()))
    );

    let unchanged = sync_mirror(&graph, &store, &options(64)).unwrap();
    assert_eq!(unchanged.mode, SyncMode::Unchanged);

    let mut rng = Rng::new(11);
    for round in 0..3 {
        mutate(&graph.read(), &ids, &mut rng);
        let stats = sync_mirror(&graph, &store, &options(64)).unwrap();
        assert_eq!(stats.mode, SyncMode::Incremental, "round {round}");
        let round_dir = dir.path().join(format!("r{round}"));
        std::fs::create_dir_all(&round_dir).unwrap();
        let full = rebuilt(&graph.read(), &round_dir);
        assert_eq!(mirror_rows(&store), mirror_rows(&full), "round {round}");
    }
}

#[test]
fn writes_made_during_a_full_copy_reach_the_mirror() {
    let dir = tempfile::tempdir().unwrap();
    let graph = open_graph(&dir.path().join("graph"));
    let ids = synthetic_graph(&graph.read(), 400, 4_000, 3);
    let store = SlowStore::open(&dir.path().join("mirror.db")).unwrap();

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let written = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let writer = {
        let (graph, stop, ids) = (Arc::clone(&graph), Arc::clone(&stop), ids.clone());
        let written = Arc::clone(&written);
        std::thread::spawn(move || {
            let mut rng = Rng::new(5);
            for _ in 0..300 {
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let (a, b) = (ids[rng.below(ids.len())], ids[rng.below(ids.len())]);
                if a != b {
                    let g = graph.read();
                    g.put_relationship_record(&edge(a, b, RelationType::RelatedTo, rng.unit()))
                        .unwrap();
                }
                written.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        })
    };
    while written.load(std::sync::atomic::Ordering::Relaxed) < 10 {
        std::thread::yield_now();
    }
    let during = sync_mirror(&graph, &store, &options(8)).unwrap();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    writer.join().unwrap();
    assert_eq!(during.mode, SyncMode::Full);
    assert!(during.changed_during_copy.is_some());

    let after = sync_mirror(&graph, &store, &options(8)).unwrap();
    assert!(matches!(
        after.mode,
        SyncMode::Incremental | SyncMode::Unchanged
    ));
    assert_eq!(
        mirror_rows(&store),
        mirror_rows(&rebuilt(&graph.read(), dir.path()))
    );
}

#[test]
fn a_scope_over_every_entity_finds_what_the_whole_mirror_finds() {
    let dir = tempfile::tempdir().unwrap();
    let graph = open_graph(&dir.path().join("graph"));
    synthetic_graph(&graph.read(), 200, 1_500, 13);
    let store = SlowStore::open(&dir.path().join("mirror.db")).unwrap();
    sync_mirror(&graph, &store, &options(64)).unwrap();

    let everything = GapScope::new(None, None, vec![], Some(10_000));
    let scoped = store.scoped_graph(&everything).unwrap();
    assert!(!scoped.report().sampled);
    assert_eq!(
        scoped.report().nodes_analyzed,
        store.entity_count().unwrap()
    );
    let sums = |t: Vec<crate::memory::slow_store::RawOpenTriad>| -> Vec<u64> {
        t.iter()
            .map(|t| (f64::from(t.ab_strength) + f64::from(t.bc_strength)).to_bits())
            .collect()
    };
    assert_eq!(
        sums(scoped.find_open_triads(0.2, 50).unwrap()),
        sums(store.find_open_triads(0.2, 50).unwrap())
    );
    assert_eq!(
        scoped.find_diamond_gaps(0.2, 50).unwrap().len(),
        store.find_diamond_gaps(0.2, 50).unwrap().len()
    );
    let hubs = |s: Vec<crate::memory::slow_store::RawStarGap>| -> BTreeSet<(String, usize, usize)> {
        s.into_iter()
            .map(|g| (g.hub, g.missing_edges, g.possible_edges))
            .collect()
    };
    assert_eq!(
        hubs(scoped.find_star_gaps(3, 0.9, 1_000).unwrap()),
        hubs(store.find_star_gaps(3, 0.9, 1_000).unwrap())
    );
    let adjacency = |mut a: HashMap<String, Vec<String>>| {
        a.values_mut().for_each(|v| v.sort());
        a.into_iter().collect::<BTreeSet<_>>()
    };
    assert_eq!(
        adjacency(scoped.get_adjacency_list(0.2).unwrap()),
        adjacency(store.get_adjacency_list(0.2).unwrap())
    );
    let config = GapDetectionConfig {
        max_gaps_per_type: 7,
        ..GapDetectionConfig::default()
    };
    let whole = GapDetector::detect(&store, &config).unwrap();
    let narrowed = GapDetector::detect(&scoped, &config).unwrap();
    // Orbit clustering visits nodes in hash order, so only the other types are compared.
    for kind in ["open_triad", "diamond_gap", "star_gap"] {
        assert_eq!(
            whole.type_counts.get(kind),
            narrowed.type_counts.get(kind),
            "{kind}"
        );
    }
    assert!(
        whole.type_counts.values().all(|&n| n <= 7),
        "{:?}",
        whole.type_counts
    );
}

#[test]
fn filters_and_samples_narrow_the_scope() {
    let dir = tempfile::tempdir().unwrap();
    let graph = open_graph(&dir.path().join("graph"));
    synthetic_graph(&graph.read(), 320, 2_000, 17);
    let store = SlowStore::open(&dir.path().join("mirror.db")).unwrap();
    sync_mirror(&graph, &store, &options(64)).unwrap();
    let names_in = |scope: GapScope| -> Vec<String> {
        let scoped = store.scoped_graph(&scope).unwrap();
        let uuids: Vec<String> = scoped
            .get_adjacency_list(0.0)
            .unwrap()
            .into_keys()
            .collect();
        let refs: Vec<&str> = uuids.iter().map(String::as_str).collect();
        scoped
            .get_entity_names(&refs)
            .unwrap()
            .into_values()
            .collect()
    };

    let files = names_in(GapScope::new(Some("FILE".into()), None, vec![], None));
    assert!(!files.is_empty());
    assert!(files.iter().all(|n| n.starts_with("file:")));

    let project = store
        .scoped_graph(&GapScope::new(None, Some("Demo".into()), vec![], None))
        .unwrap();
    assert!(project.report().nodes_in_scope > 0);
    let tagged = names_in(GapScope::new(None, Some("demo".into()), vec![], None));
    assert!(tagged.iter().all(|n| n != "project:demo"));

    let people = store
        .scoped_graph(&GapScope::new(None, None, vec!["person".into()], None))
        .unwrap();
    assert!(people.report().nodes_in_scope > 0 && people.report().nodes_in_scope < 321);

    let sample = store
        .scoped_graph(&GapScope::new(None, None, vec![], Some(25)))
        .unwrap();
    assert!(sample.report().sampled);
    assert_eq!(sample.report().nodes_analyzed, 25);
    assert_eq!(
        sample.report().nodes_in_scope,
        store.entity_count().unwrap()
    );
}
