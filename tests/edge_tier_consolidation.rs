//! Edge-tier consolidation on a batch-ingested store, ported from upstream #461.
//!
//! An import, an eval run or a seeded deployment ingests its corpus in one pass, so every
//! edge is minutes old when the pass ends. Upstream gated promotion on time spent in a tier
//! and added a count of distinct attesting episodes so such a store could consolidate at all.
//! veld has no clock gate: an edge promotes when its strength crosses its tier's threshold,
//! and only a source new to the edge moves strength (`RelationshipEdge::strengthen_from`).
//! These tests pin, on a real store through `add_relationship`, that a batch ingest
//! consolidates exactly the edges another episode corroborated and nothing else.
//!
//! Upstream measured with the GLiNER weights; this port uses the rule-based extractor, which
//! needs no model. The invariants hold for any entity population, and the gate corpus still
//! decides how often a pair recurs.
//!
//! Upstream's second test, that corroborated edges are shielded from the lazy prune, is not
//! ported: its premise is that L2 prunes earlier than L1 at about the same decay rate, while
//! veld decays each tier at its own rate (`L1_DECAY_PER_HOUR` reaches the L1 floor at 10
//! days, `L2_DECAY_PER_DAY` the L2 floor at 90).

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use tempfile::TempDir;
use veld::constants::{
    L1_INITIAL_WEIGHT, L1_PROMOTION_THRESHOLD, LTP_LEARNING_RATE, TIER_CO_ACCESS_BOOST,
};
use veld::embeddings::ner::{NerConfig, NeuralNer};
use veld::graph_memory::{
    EdgeSource, EdgeTier, EntityLabel, EntityNode, GraphMemory, LtpStatus, PiiClassification,
    RelationType, RelationshipEdge,
};
use veld::uuid::Uuid;

fn entity(name: &str) -> EntityNode {
    let now = chrono::Utc::now();
    EntityNode {
        uuid: Uuid::new_v4(),
        name: name.to_string(),
        labels: vec![EntityLabel::Concept],
        created_at: now,
        last_seen_at: now,
        mention_count: 1,
        summary: String::new(),
        attributes: HashMap::new(),
        name_embedding: None,
        salience: 0.5,
        is_proper_noun: true,
        pii_classification: PiiClassification::default(),
    }
}

/// A co-occurrence edge as `MemorySystem` builds one for a memory: full L1 weight, the
/// episode named, and no strengthening source recorded for the episode that creates it.
fn cooccurrence(from: Uuid, to: Uuid, episode: Uuid) -> RelationshipEdge {
    let now = chrono::Utc::now();
    RelationshipEdge {
        uuid: Uuid::new_v4(),
        from_entity: from,
        to_entity: to,
        relation_type: RelationType::CoOccurs,
        strength: L1_INITIAL_WEIGHT,
        created_at: now,
        valid_at: now,
        invalidated_at: None,
        source_episode_id: Some(episode),
        context: String::new(),
        last_activated: now,
        activation_count: 1,
        ltp_status: LtpStatus::None,
        tier: EdgeTier::L1Working,
        activation_timestamps: None,
        entity_confidence: None,
        created_by: EdgeSource::CoOccurrence,
        forward_strength: L1_INITIAL_WEIGHT,
        backward_strength: L1_INITIAL_WEIGHT,
        strengthening_sources: BTreeMap::new(),
        last_corroborated: None,
    }
}

/// The LoCoMo gate corpus: 629 conversational turns, the text the CI recall gate runs on.
fn gate_corpus() -> Vec<String> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/recall/corpora/locomo-gate.jsonl");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read corpus {}: {e}", path.display()));
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let item: serde_json::Value = serde_json::from_str(line).expect("corpus line is JSON");
            item["content"]
                .as_str()
                .expect("corpus item content")
                .to_string()
        })
        .collect()
}

/// One corpus item is one episode. Every unordered pair of distinct entities in it is
/// attested once, as `MemorySystem` attests a memory's co-occurrences.
fn ingest(graph: &GraphMemory, corpus: &[String]) {
    let ner = NeuralNer::new_fallback(NerConfig::default());
    let mut entities: HashMap<String, Uuid> = HashMap::new();
    for content in corpus {
        let episode = Uuid::new_v4();
        let mut ids = Vec::new();
        for mention in ner.extract(content).unwrap_or_default() {
            let name = mention.text.trim().to_string();
            if name.is_empty() {
                continue;
            }
            let id = match entities.get(&name) {
                Some(id) => *id,
                None => {
                    let id = graph.add_entity(entity(&name)).expect("add entity");
                    entities.insert(name, id);
                    id
                }
            };
            ids.push(id);
        }
        // add_entity resolves case and stem variants to one node, so two mentions can be
        // one entity; a pair is attested once per episode however many mentions name it.
        ids.sort();
        ids.dedup();
        for (i, from) in ids.iter().enumerate() {
            for to in &ids[i + 1..] {
                graph
                    .add_relationship(cooccurrence(*from, *to, episode))
                    .expect("add relationship");
            }
        }
    }
}

#[test]
fn batch_ingest_consolidates_by_evidence_not_by_clock() {
    let one_corroboration =
        L1_INITIAL_WEIGHT + (LTP_LEARNING_RATE + TIER_CO_ACCESS_BOOST) * (1.0 - L1_INITIAL_WEIGHT);
    assert!(
        one_corroboration >= L1_PROMOTION_THRESHOLD,
        "one corroborating episode lifts a full-weight L1 edge to {one_corroboration}, under \
         the {L1_PROMOTION_THRESHOLD} promotion threshold; the census below assumes it crosses"
    );

    let dir = TempDir::new().expect("temp dir");
    let graph = GraphMemory::new(dir.path(), None).expect("open graph memory");
    let corpus = gate_corpus();
    assert!(
        corpus.len() > 100,
        "expected the gate corpus, got {} items",
        corpus.len()
    );
    ingest(&graph, &corpus);

    let edges = graph.get_all_relationships().expect("all edges");
    assert!(!edges.is_empty(), "ingest produced no edges");

    let mut by_tier: BTreeMap<String, usize> = BTreeMap::new();
    let mut corroborated = 0;
    for edge in &edges {
        *by_tier.entry(format!("{:?}", edge.tier)).or_default() += 1;
        let sources = edge.distinct_sources();
        if sources > 0 {
            corroborated += 1;
        }
        match edge.tier {
            EdgeTier::L1Working => assert_eq!(
                sources, 0,
                "edge {} was corroborated by {sources} other episode(s) and is still at L1",
                edge.uuid
            ),
            EdgeTier::L2Episodic => assert!(
                sources >= 1,
                "edge {} reached L2 with no episode but its creator behind it",
                edge.uuid
            ),
            EdgeTier::L3Semantic => assert!(
                sources >= 2,
                "edge {} reached L3 on {sources} corroborating episode(s); one moves an edge \
                 a single tier",
                edge.uuid
            ),
        }
    }
    println!(
        "{} episodes, {} edges, {corroborated} corroborated, by tier {by_tier:?}",
        corpus.len(),
        edges.len()
    );

    assert!(
        corroborated > 0,
        "no entity pair recurred across episodes, so the invariants above held vacuously"
    );
    assert!(
        by_tier.get("L1Working").copied().unwrap_or(0) > 0,
        "every edge left L1, though a pair seen in one episode has nothing to corroborate it"
    );
}

#[test]
fn each_tier_takes_corroboration_from_distinct_episodes() {
    let dir = TempDir::new().expect("temp dir");
    let graph = GraphMemory::new(dir.path(), None).expect("open graph memory");
    let a = graph.add_entity(entity("Harbour")).expect("add entity");
    let b = graph.add_entity(entity("Lighthouse")).expect("add entity");

    let attest = |episode: Uuid| {
        let id = graph
            .add_relationship(cooccurrence(a, b, episode))
            .expect("add relationship");
        graph
            .get_all_relationships()
            .expect("all edges")
            .into_iter()
            .find(|edge| edge.uuid == id)
            .expect("the attested edge is stored")
    };

    let created = attest(Uuid::new_v4());
    assert_eq!(created.tier, EdgeTier::L1Working);
    assert_eq!(created.distinct_sources(), 0);

    let second = Uuid::new_v4();
    let corroborated = attest(second);
    assert_eq!(corroborated.uuid, created.uuid, "a pair is one synapse");
    assert_eq!(corroborated.tier, EdgeTier::L2Episodic);
    assert_eq!(corroborated.distinct_sources(), 1);

    let repeated = attest(second);
    assert_eq!(
        repeated.tier,
        EdgeTier::L2Episodic,
        "a repeat is not corroboration"
    );
    assert_eq!(repeated.strength, corroborated.strength);
    assert_eq!(repeated.distinct_sources(), 1);
    assert_eq!(
        repeated.strengthening_sources[&format!("episode:{second}")],
        2
    );

    let third = attest(Uuid::new_v4());
    assert_eq!(third.tier, EdgeTier::L2Episodic);
    assert_eq!(third.distinct_sources(), 2);

    let fourth = attest(Uuid::new_v4());
    assert_eq!(fourth.tier, EdgeTier::L3Semantic);
    assert_eq!(fourth.distinct_sources(), 3);
}
