//! Writer latency while a gap analysis runs, before and after the scale work.
//!
//! A synthetic graph is built in a scratch directory. One thread writes
//! memories the way remember does (entities, an episode and co-occurrence
//! edges under the graph's read lock, then a projection row in the slow
//! store) every `GAP_BENCH_INTERVAL_MS`; another takes the graph's write
//! lock every 200 ms for one edge, as edge proposals and entity deletes do.
//! Latencies are kept per phase:
//!
//! - `baseline`: no analysis.
//! - `legacy`: the path before this change: the whole graph cloned under one
//!   read lock, the previous mirror rebuild (`rebuild_mirror_as_before`),
//!   detection unpaced at normal priority. Detection itself is today's.
//! - `worker_full`: the gap pipeline on the gap worker, first run (full copy).
//! - `worker_incremental`: the same, next run (incremental sync).
//!
//! Run: `cargo test --release --lib gap_analysis_writer_latency -- --ignored --nocapture`.
//! `--features stress` sizes the graph like the largest user graph (88k
//! entities, 1.03M edges); `GAP_BENCH_ENTITIES` / `_EDGES` override.

use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

use super::gap_scale_tests::{edge, entity, options, synthetic_graph, Rng};
use crate::graph_memory::{EntityLabel, EpisodeSource, EpisodicNode, GraphMemory, RelationType};
use crate::memory::gap_topology::{
    run_gap_analysis, worker, GapDetectionConfig, GapDetector, GapRunParams, GapScope,
    GapWorkerConfig,
};
use crate::memory::slow_store::SlowStore;

const PHASES: [&str; 4] = ["baseline", "legacy", "worker_full", "worker_incremental"];
const BASELINE: u8 = 0;
const LEGACY: u8 = 1;
const WORKER_FULL: u8 = 2;
const WORKER_INCREMENTAL: u8 = 3;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn sizes() -> (usize, usize) {
    let (entities, edges) = if cfg!(feature = "stress") {
        (88_000, 1_030_000)
    } else {
        (20_000, 235_000)
    };
    (
        env_usize("GAP_BENCH_ENTITIES", entities),
        env_usize("GAP_BENCH_EDGES", edges),
    )
}

#[derive(Default)]
struct Latencies {
    memory: [Vec<Duration>; 4],
    /// The projection write alone.
    projection: [Vec<Duration>; 4],
    exclusive: [Vec<Duration>; 4],
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// One memory written the way remember writes it to the graph and the
/// projection; returns how long the projection write took.
fn write_memory(
    graph: &RwLock<GraphMemory>,
    store: &SlowStore,
    names: &[String],
    rng: &mut Rng,
    seq: u64,
) -> Duration {
    {
        let g = graph.read();
        let mut ids: Vec<Uuid> = Vec::with_capacity(4);
        for i in 0..4 {
            let name = if i < 2 {
                names[rng.below(names.len())].clone()
            } else {
                format!("bench-{seq}-{i}")
            };
            ids.push(
                g.add_entity(entity(&name, EntityLabel::Concept, 0.5))
                    .unwrap(),
            );
        }
        let now = chrono::Utc::now();
        let episode = g
            .add_episode(EpisodicNode {
                uuid: Uuid::new_v4(),
                name: format!("memory {seq}"),
                content: "bench memory".into(),
                valid_at: now,
                created_at: now,
                entity_refs: ids.clone(),
                source: EpisodeSource::Message,
                metadata: HashMap::new(),
            })
            .unwrap();
        let mut edges = Vec::new();
        for a in 0..ids.len() {
            for b in a + 1..ids.len() {
                let mut e = edge(ids[a], ids[b], RelationType::RelatedTo, 0.3);
                e.source_episode_id = Some(episode);
                edges.push(e);
            }
        }
        g.add_relationships(edges).unwrap();
    }
    let start = Instant::now();
    store
        .upsert_memory("bench", &format!("m{seq}"), seq, &[7u8; 1024], 0.5)
        .unwrap();
    start.elapsed()
}

/// The gap analysis as it ran before: the whole graph cloned under one read
/// lock, the previous rebuild, detection on a normal-priority thread.
fn legacy_analysis(graph: &RwLock<GraphMemory>, store: &SlowStore) -> (u64, u64) {
    let start = Instant::now();
    let (entities, edges) = {
        let g = graph.read();
        (
            g.get_all_entities().unwrap(),
            g.get_all_relationships().unwrap(),
        )
    };
    store.rebuild_mirror_as_before(&entities, &edges).unwrap();
    drop((entities, edges));
    let sync_ms = start.elapsed().as_millis() as u64;
    GapDetector::detect(store, &GapDetectionConfig::default()).unwrap();
    (sync_ms, start.elapsed().as_millis() as u64)
}

#[test]
#[ignore = "measurement: cargo test --release --lib gap_analysis_writer_latency -- --ignored --nocapture"]
fn gap_analysis_writer_latency() {
    let (n_entities, n_edges) = sizes();
    let interval = Duration::from_millis(env_usize("GAP_BENCH_INTERVAL_MS", 5) as u64);
    let baseline_secs = env_usize("GAP_BENCH_BASELINE_SECS", 15) as u64;
    let dir = tempfile::tempdir().unwrap();
    let graph = Arc::new(RwLock::new(
        GraphMemory::new(&dir.path().join("graph"), None).unwrap(),
    ));
    let built = Instant::now();
    let ids = synthetic_graph(&graph.read(), n_entities, n_edges, 42);
    let names: Vec<String> = (0..n_entities)
        .filter(|i| i % 16 != 0)
        .take(5_000)
        .map(|i| format!("entity-{i}"))
        .collect();
    println!(
        "graph: {n_entities} entities, {} relationship rows, built in {:.1}s",
        graph.read().get_stats().unwrap().relationship_count,
        built.elapsed().as_secs_f64()
    );

    let open = |name: &str| Arc::new(SlowStore::open(&dir.path().join(name)).unwrap());
    let current = Arc::new(Mutex::new(open("baseline.db")));
    let phase = Arc::new(AtomicU8::new(BASELINE));
    let done = Arc::new(AtomicBool::new(false));
    let latencies = Arc::new(Mutex::new(Latencies::default()));

    let memory_writer = {
        let (graph, current, phase, done, latencies, names) = (
            Arc::clone(&graph),
            Arc::clone(&current),
            Arc::clone(&phase),
            Arc::clone(&done),
            Arc::clone(&latencies),
            names.clone(),
        );
        std::thread::spawn(move || {
            let mut rng = Rng::new(9);
            let mut seq = 0u64;
            let mut next = Instant::now();
            while !done.load(Ordering::Relaxed) {
                let store = Arc::clone(&current.lock());
                let at = phase.load(Ordering::Relaxed) as usize;
                let start = Instant::now();
                let projection = write_memory(&graph, &store, &names, &mut rng, seq);
                let mut lat = latencies.lock();
                lat.memory[at].push(start.elapsed());
                lat.projection[at].push(projection);
                seq += 1;
                next += interval;
                if let Some(wait) = next.checked_duration_since(Instant::now()) {
                    std::thread::sleep(wait);
                } else {
                    next = Instant::now();
                }
            }
        })
    };
    let exclusive_writer = {
        let (graph, phase, done, latencies, ids) = (
            Arc::clone(&graph),
            Arc::clone(&phase),
            Arc::clone(&done),
            Arc::clone(&latencies),
            ids.clone(),
        );
        std::thread::spawn(move || {
            let mut rng = Rng::new(21);
            while !done.load(Ordering::Relaxed) {
                let (a, b) = (ids[rng.below(ids.len())], ids[rng.below(ids.len())]);
                let at = phase.load(Ordering::Relaxed) as usize;
                let start = Instant::now();
                graph
                    .write()
                    .add_relationships(vec![edge(a, b, RelationType::RelatedTo, 0.4)])
                    .unwrap();
                latencies.lock().exclusive[at].push(start.elapsed());
                std::thread::sleep(Duration::from_millis(200));
            }
        })
    };

    let mut timings: Vec<(&str, u64, u64)> = Vec::new();
    std::thread::sleep(Duration::from_secs(baseline_secs));

    let store = open("legacy.db");
    *current.lock() = Arc::clone(&store);
    phase.store(LEGACY, Ordering::Relaxed);
    let (sync_ms, total_ms) = {
        let graph = Arc::clone(&graph);
        std::thread::spawn(move || legacy_analysis(&graph, &store))
            .join()
            .unwrap()
    };
    timings.push(("legacy", sync_ms, total_ms));
    phase.store(BASELINE, Ordering::Relaxed);
    std::thread::sleep(Duration::from_secs(baseline_secs / 3));

    let store = open("worker.db");
    *current.lock() = Arc::clone(&store);
    let defaults = GapWorkerConfig::default();
    let config = GapWorkerConfig {
        nice: env_usize("GAP_BENCH_NICE", defaults.nice as usize) as i32,
        chunk_rows: env_usize("GAP_BENCH_CHUNK_ROWS", defaults.chunk_rows),
        cpu_pct: env_usize("GAP_BENCH_CPU_PCT", defaults.cpu_pct as usize) as u8,
    };
    println!("worker: {config:?}");
    let params = GapRunParams {
        config: GapDetectionConfig::default(),
        scope: GapScope::default(),
        mirror: options(config.chunk_rows),
    };
    for (name, at) in [
        ("worker_full", WORKER_FULL),
        ("worker_incremental", WORKER_INCREMENTAL),
    ] {
        phase.store(at, Ordering::Relaxed);
        let (tx, rx) = std::sync::mpsc::channel();
        let (graph, store, params) = (Arc::clone(&graph), Arc::clone(&store), params.clone());
        worker::submit(&config, move || {
            let _ = tx.send(run_gap_analysis(&graph, &store, &params).unwrap());
        })
        .unwrap();
        let run = rx.recv().unwrap();
        println!(
            "{name}: mirror {:?}, {} rows written, changed during copy {:?}",
            run.mirror.mode, run.mirror.rows_written, run.mirror.changed_during_copy
        );
        timings.push((name, run.mirror.duration_ms, run.duration_ms));
        phase.store(BASELINE, Ordering::Relaxed);
        std::thread::sleep(Duration::from_secs(baseline_secs / 3));
    }

    done.store(true, Ordering::Relaxed);
    memory_writer.join().unwrap();
    exclusive_writer.join().unwrap();

    let mut guard = latencies.lock();
    let lat = &mut *guard;
    for v in lat
        .memory
        .iter_mut()
        .chain(lat.projection.iter_mut())
        .chain(lat.exclusive.iter_mut())
    {
        v.sort();
    }
    println!(
        "\n{:<20} {:>7} {:>9} {:>9} {:>9} {:>9} | {:>5} {:>9} {:>9} | {:>9} {:>9}",
        "phase",
        "writes",
        "p50 ms",
        "p99 ms",
        "max ms",
        "proj p99",
        "excl",
        "p50 ms",
        "p99 ms",
        "sync ms",
        "total ms"
    );
    for (i, name) in PHASES.iter().enumerate() {
        let (m, x) = (&lat.memory[i], &lat.exclusive[i]);
        let (sync, total) = timings
            .iter()
            .find(|t| t.0 == *name)
            .map_or((String::from("-"), String::from("-")), |t| {
                (t.1.to_string(), t.2.to_string())
            });
        println!(
            "{:<20} {:>7} {:>9.2} {:>9.2} {:>9.2} {:>9.2} | {:>5} {:>9.2} {:>9.2} | {:>9} {:>9}",
            name,
            m.len(),
            ms(percentile(m, 0.5)),
            ms(percentile(m, 0.99)),
            ms(m.last().copied().unwrap_or_default()),
            ms(percentile(&lat.projection[i], 0.99)),
            x.len(),
            ms(percentile(x, 0.5)),
            ms(percentile(x, 0.99)),
            sync,
            total,
        );
    }
    let base_p99 = percentile(&lat.memory[BASELINE as usize], 0.99);
    let max_ratio: f64 = std::env::var("GAP_BENCH_MAX_P99_RATIO")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3.0);
    for at in [LEGACY, WORKER_FULL, WORKER_INCREMENTAL] {
        let ratio = ms(percentile(&lat.memory[at as usize], 0.99)) / ms(base_p99).max(1e-3);
        println!("writer p99 ratio {}: {ratio:.2}", PHASES[at as usize]);
        if at != LEGACY {
            assert!(
                ratio <= max_ratio,
                "{}: writer p99 is {ratio:.2}x the baseline's, above {max_ratio}",
                PHASES[at as usize]
            );
        }
    }
}
