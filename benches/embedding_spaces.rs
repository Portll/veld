//! What each embedding space costs to write, search and hold.
//!
//! Step 12 of docs/EXECUTION-embedding-spaces-2026-10-19.md: the baseline recorded before
//! deferred embedding (step 9) lands, re-run after every later step, so each step has a
//! before/after row. Four measurements, each a row in one JSON report:
//!
//! - **search**: vector-leg latency (p50/p95/p99) over one space and over two, at each corpus
//!   size: one index search per space, in sequence.
//! - **query**: the query path per space, embedding included (the embedder's simulated cost
//!   per call and per text), over one space and over two: in sequence, as recall paid it
//!   before step 10, and at once, as it does now: both embeddings together, the second
//!   space's search on its own thread beside the primary's. Step 10 is done when the
//!   concurrent row's p95 is within 1.3x of the one-space row's at 100 000 memories. In
//!   recall the second space's search also overlaps graph traversal and BM25, which this row
//!   leaves out, so it bounds the overlap from below.
//! - **footprint**: bytes a space holds as full vectors and as product-quantized codes, and the
//!   process RSS growth while its index is built; per size, what two spaces (the second as
//!   `pq`, the default codec since step 11) hold against one. Step 11 is done when that is
//!   under 1.3x at 100 000 memories.
//! - **quality**: recall@10 of each codec's search against exact search, the `pq` index's
//!   candidates reranked against the exact vectors as the server does (here read from
//!   memory, there from the records).
//! - **write**: inline write cost through the retrieval engine, embedding included, with an
//!   embedder that sleeps a configurable round trip per call and cost per text.
//! - **drain**: texts per second per batch size against a real embedder host, when one is set.
//!
//! The vector legs are measured on the index directly: the engine adds a mapping lookup and a
//! chunk merge, which do not depend on the number of spaces. The LoCoMo gate under each space
//! configuration runs in CI (`.github/workflows/recall.yml`), not here.
//!
//! Run: `cargo bench --bench embedding_spaces`
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `VELD_BENCH_SPACE_SIZES` | `10000` | corpus sizes; the plan's rows are `10000,100000` |
//! | `VELD_BENCH_SPACE_QUERIES` | `200` | queries per search configuration |
//! | `VELD_BENCH_WRITE_COUNT` | `300` | memories per write measurement |
//! | `VELD_BENCH_EMBED_CALL_US` | `3000` | simulated embedder round trip per call |
//! | `VELD_BENCH_EMBED_TEXT_US` | `200` | simulated embedder cost per text |
//! | `VELD_BENCH_EMBEDDER_URL` | unset | OpenAI-compatible host for the drain rows |
//! | `VELD_BENCH_EMBEDDER_MODEL` | `text-embedding-nomic-embed-text-v1.5` | its model |
//! | `VELD_BENCH_SPACE_OUT` | `evaluations/spaces/<utc>.json` | report path |

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use veld::embeddings::competitive::CompetitiveEmbedder;
use veld::embeddings::http_embedder::{HttpEmbedder, HttpEmbedderConfig};
use veld::embeddings::space::SpaceId;
use veld::embeddings::Embedder;
use veld::memory::retrieval::RetrievalEngine;
use veld::memory::types::{Experience, Memory, MemoryId};
use veld::storage::PrimaryMemoryStore;
use veld::vector_db::quantized::QuantizedIndex;
use veld::vector_db::vamana::{VamanaConfig, VamanaIndex};

const PRIMARY_DIM: usize = 768;
const SECONDARY_DIM: usize = 384;
const CLUSTERS: usize = 64;
const RECALL_LIMIT: usize = 10;
const DRAIN_BATCHES: [usize; 5] = [1, 8, 32, 64, 128];
const DRAIN_TEXTS: usize = 256;

#[derive(Serialize)]
struct Report {
    generated_at: String,
    git_sha: Option<String>,
    sizes: Vec<usize>,
    rows: Vec<Row>,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Row {
    Search {
        memories: usize,
        spaces: Vec<String>,
        codec: &'static str,
        queries: usize,
        p50_us: u64,
        p95_us: u64,
        p99_us: u64,
    },
    Query {
        memories: usize,
        mode: &'static str,
        embed_us: u64,
        queries: usize,
        p50_us: u64,
        p95_us: u64,
        p99_us: u64,
        p95_over_one_space: f64,
    },
    Footprint {
        memories: usize,
        space: String,
        codec: &'static str,
        bytes: usize,
        rss_growth_bytes: Option<u64>,
        build_ms: u64,
    },
    TwoSpaces {
        memories: usize,
        one_space_bytes: usize,
        two_spaces_bytes: usize,
        bytes_ratio: f64,
        rss_ratio: Option<f64>,
    },
    Quality {
        memories: usize,
        space: String,
        codec: &'static str,
        queries: usize,
        recall_at_10: f64,
    },
    Write {
        mode: &'static str,
        spaces: Vec<String>,
        memories: usize,
        per_memory_us: u64,
        embed_call_us: u64,
        embed_text_us: u64,
    },
    Drain {
        host: String,
        model: String,
        batch_size: usize,
        texts: usize,
        texts_per_sec: f64,
    },
    Skipped {
        what: String,
        why: String,
    },
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn env_sizes() -> Vec<usize> {
    std::env::var("VELD_BENCH_SPACE_SIZES")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|s| s.trim().parse().ok())
                .filter(|n| *n > 0)
                .collect::<Vec<usize>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![10_000])
}

fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// Unit vectors drawn around `CLUSTERS` centres: embeddings cluster by topic, and an index
/// built over uniform noise measures a corpus nobody has.
fn clustered(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centres: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|_| normalized((0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()))
        .collect();
    (0..n)
        .map(|i| {
            let centre = &centres[i % CLUSTERS];
            normalized(
                centre
                    .iter()
                    .map(|c| c + rng.gen_range(-0.35..0.35))
                    .collect(),
            )
        })
        .collect()
}

fn queries_near(corpus: &[Vec<f32>], count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let base = &corpus[rng.gen_range(0..corpus.len())];
            normalized(base.iter().map(|x| x + rng.gen_range(-0.1..0.1)).collect())
        })
        .collect()
}

/// Queries the quality rows score against exact search, which reads every vector per query.
const QUALITY_QUERIES: usize = 50;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na * nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

/// The `k` vectors nearest `query` by exact cosine.
fn exact_top(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, cosine(query, v)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn percentile(sorted: &[Duration], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[rank].as_micros() as u64
}

fn rss_bytes() -> Option<u64> {
    let pid = sysinfo::get_current_pid().ok()?;
    let mut system = sysinfo::System::new();
    system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    system.process(pid).map(|p| p.memory())
}

fn vamana(dim: usize, vectors: Vec<Vec<f32>>) -> Result<VamanaIndex> {
    let mut index = VamanaIndex::new(VamanaConfig {
        dimension: dim,
        max_degree: 32,
        search_list_size: 100,
        alpha: 1.2,
        use_mmap: false,
        ..Default::default()
    })?;
    index.build(vectors)?;
    Ok(index)
}

/// The candidate count recall asks an index for, per space.
fn candidates() -> usize {
    RECALL_LIMIT * veld::constants::VECTOR_SEARCH_CANDIDATE_MULTIPLIER * 2
}

fn timed_searches(
    queries: &[Vec<f32>],
    mut search: impl FnMut(usize) -> Result<()>,
) -> Result<Vec<Duration>> {
    let mut samples = Vec::with_capacity(queries.len());
    for i in 0..queries.len() {
        let start = Instant::now();
        search(i)?;
        samples.push(start.elapsed());
    }
    samples.sort();
    Ok(samples)
}

fn search_row(
    memories: usize,
    spaces: &[&SpaceId],
    codec: &'static str,
    samples: &[Duration],
) -> Row {
    Row::Search {
        memories,
        spaces: spaces.iter().map(|s| s.to_string()).collect(),
        codec,
        queries: samples.len(),
        p50_us: percentile(samples, 0.50),
        p95_us: percentile(samples, 0.95),
        p99_us: percentile(samples, 0.99),
    }
}

fn query_row(
    memories: usize,
    mode: &'static str,
    embed: Duration,
    samples: &[Duration],
    one_space_p95: u64,
) -> Row {
    let p95_us = percentile(samples, 0.95);
    Row::Query {
        memories,
        mode,
        embed_us: embed.as_micros() as u64,
        queries: samples.len(),
        p50_us: percentile(samples, 0.50),
        p95_us,
        p99_us: percentile(samples, 0.99),
        p95_over_one_space: p95_us as f64 / one_space_p95.max(1) as f64,
    }
}

fn measure_search_and_footprint(
    sizes: &[usize],
    query_count: usize,
    embed: Duration,
) -> Result<Vec<Row>> {
    let primary = SpaceId::new("bench-primary", PRIMARY_DIM);
    let secondary = SpaceId::new("bench-secondary", SECONDARY_DIM);
    let mut rows = Vec::new();
    for &n in sizes {
        let primary_vectors = clustered(n, PRIMARY_DIM, 1);
        let secondary_vectors = clustered(n, SECONDARY_DIM, 2);
        let primary_queries = queries_near(&primary_vectors, query_count, 3);
        let secondary_queries = queries_near(&secondary_vectors, query_count, 4);

        let rss_before = rss_bytes();
        let start = Instant::now();
        let primary_index = vamana(PRIMARY_DIM, primary_vectors.clone())?;
        let primary_build = start.elapsed();
        let rss_primary = rss_bytes();
        rows.push(Row::Footprint {
            memories: n,
            space: primary.to_string(),
            codec: "f32",
            bytes: n * PRIMARY_DIM * 4,
            rss_growth_bytes: rss_before
                .zip(rss_primary)
                .map(|(a, b)| b.saturating_sub(a)),
            build_ms: primary_build.as_millis() as u64,
        });

        let start = Instant::now();
        let secondary_index = vamana(SECONDARY_DIM, secondary_vectors.clone())?;
        let secondary_build = start.elapsed();
        let rss_secondary = rss_bytes();
        rows.push(Row::Footprint {
            memories: n,
            space: secondary.to_string(),
            codec: "f32",
            bytes: n * SECONDARY_DIM * 4,
            rss_growth_bytes: rss_primary
                .zip(rss_secondary)
                .map(|(a, b)| b.saturating_sub(a)),
            build_ms: secondary_build.as_millis() as u64,
        });

        let k = candidates();
        let one = timed_searches(&primary_queries, |i| {
            primary_index.search(&primary_queries[i], k).map(|_| ())
        })?;
        rows.push(search_row(n, &[&primary], "f32", &one));

        let two = timed_searches(&primary_queries, |i| {
            primary_index.search(&primary_queries[i], k)?;
            secondary_index.search(&secondary_queries[i], k).map(|_| ())
        })?;
        rows.push(search_row(n, &[&primary, &secondary], "f32", &two));

        let embed_query = || std::thread::sleep(embed);
        let one_path = timed_searches(&primary_queries, |i| {
            embed_query();
            primary_index.search(&primary_queries[i], k).map(|_| ())
        })?;
        let one_p95 = percentile(&one_path, 0.95);
        rows.push(query_row(n, "one-space", embed, &one_path, one_p95));
        let sequential = timed_searches(&primary_queries, |i| {
            embed_query();
            embed_query();
            primary_index.search(&primary_queries[i], k)?;
            secondary_index.search(&secondary_queries[i], k).map(|_| ())
        })?;
        rows.push(query_row(n, "two-sequential", embed, &sequential, one_p95));
        let concurrent = timed_searches(&primary_queries, |i| {
            std::thread::scope(|scope| {
                let secondary_embed = scope.spawn(embed_query);
                embed_query();
                secondary_embed
                    .join()
                    .map_err(|_| anyhow::anyhow!("the embedding thread panicked"))
            })?;
            std::thread::scope(|scope| {
                let leg =
                    scope.spawn(|| secondary_index.search(&secondary_queries[i], k).map(|_| ()));
                primary_index.search(&primary_queries[i], k)?;
                leg.join()
                    .map_err(|_| anyhow::anyhow!("the leg thread panicked"))?
            })
        })?;
        rows.push(query_row(n, "two-concurrent", embed, &concurrent, one_p95));

        // The secondary space as the default codec holds it: product-quantized codes searched
        // flat, each search's best candidates reranked against the exact vectors.
        let rss_before_pq = rss_bytes();
        let start = Instant::now();
        let mut pq = QuantizedIndex::new(SECONDARY_DIM);
        pq.build(secondary_vectors.clone())?;
        let pq_build = start.elapsed();
        let rss_pq = rss_before_pq
            .zip(rss_bytes())
            .map(|(a, b)| b.saturating_sub(a));
        rows.push(Row::Footprint {
            memories: n,
            space: secondary.to_string(),
            codec: "pq",
            bytes: pq.memory_bytes(),
            rss_growth_bytes: rss_pq,
            build_ms: pq_build.as_millis() as u64,
        });
        let one_space_bytes = n * PRIMARY_DIM * 4;
        let two_spaces_bytes = one_space_bytes + pq.memory_bytes();
        let rss_one = rss_before
            .zip(rss_primary)
            .map(|(a, b)| b.saturating_sub(a));
        rows.push(Row::TwoSpaces {
            memories: n,
            one_space_bytes,
            two_spaces_bytes,
            bytes_ratio: two_spaces_bytes as f64 / one_space_bytes.max(1) as f64,
            rss_ratio: rss_one
                .zip(rss_pq)
                .filter(|(one, _)| *one > 0)
                .map(|(one, pq)| (one + pq) as f64 / one as f64),
        });

        let leg_limit = RECALL_LIMIT * 8;
        let pq_search = |q: &[f32]| -> Result<Vec<u32>> {
            let mut hits: Vec<(u32, f32)> = pq
                .search(q, leg_limit * 2)?
                .into_iter()
                .map(|(id, _)| (id, cosine(q, &secondary_vectors[id as usize])))
                .collect();
            hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            hits.truncate(leg_limit);
            Ok(hits.into_iter().map(|(id, _)| id).collect())
        };
        let two_pq = timed_searches(&primary_queries, |i| {
            primary_index.search(&primary_queries[i], k)?;
            pq_search(&secondary_queries[i]).map(|_| ())
        })?;
        rows.push(search_row(n, &[&primary, &secondary], "pq", &two_pq));

        let probes = &secondary_queries[..secondary_queries.len().min(QUALITY_QUERIES)];
        let mut found_f32 = 0usize;
        let mut found_pq = 0usize;
        for q in probes {
            let truth = exact_top(&secondary_vectors, q, 10);
            let graph: Vec<u32> = secondary_index
                .search(q, k)?
                .into_iter()
                .take(10)
                .map(|(id, _)| id)
                .collect();
            found_f32 += graph.iter().filter(|id| truth.contains(id)).count();
            found_pq += pq_search(q)?
                .into_iter()
                .take(10)
                .filter(|id| truth.contains(id))
                .count();
        }
        for (codec, found) in [("f32", found_f32), ("pq", found_pq)] {
            rows.push(Row::Quality {
                memories: n,
                space: secondary.to_string(),
                codec,
                queries: probes.len(),
                recall_at_10: found as f64 / (probes.len() * 10).max(1) as f64,
            });
        }
    }
    Ok(rows)
}

/// An embedder that costs what a remote host costs: a round trip per call and a little per
/// text, so batching shows. Vectors are deterministic per text.
struct PacedEmbedder {
    id: String,
    model: &'static str,
    dim: usize,
    per_call: Duration,
    per_text: Duration,
}

impl PacedEmbedder {
    fn vector(&self, text: &str) -> Vec<f32> {
        let seed = text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
        let mut rng = StdRng::seed_from_u64(seed);
        normalized((0..self.dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
    }
}

impl Embedder for PacedEmbedder {
    fn id(&self) -> &str {
        &self.id
    }
    fn model_id(&self) -> &str {
        self.model
    }
    fn dimension(&self) -> usize {
        self.dim
    }
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        std::thread::sleep(self.per_call + self.per_text);
        Ok(self.vector(text))
    }
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        std::thread::sleep(self.per_call + self.per_text * texts.len() as u32);
        Ok(texts.iter().map(|t| self.vector(t)).collect())
    }
}

#[cfg(feature = "rocksdb")]
fn open_store(dir: &Path) -> Result<Arc<dyn PrimaryMemoryStore>> {
    let storage = veld::memory::storage::MemoryStorage::new(dir, None)?;
    Ok(Arc::new(
        veld::storage::legacy_rocksdb::RocksDbPrimaryMemoryStore::from_arc(Arc::new(storage)),
    ))
}

#[cfg(all(not(feature = "rocksdb"), feature = "storage-redb"))]
fn open_store(dir: &Path) -> Result<Arc<dyn PrimaryMemoryStore>> {
    Ok(Arc::new(veld::storage::redb::RedbPrimaryMemoryStore::open(
        dir,
    )?))
}

fn measure_inline_writes(count: usize, call: Duration, text: Duration) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    for two_spaces in [false, true] {
        let dir = tempfile::tempdir()?;
        let store = open_store(dir.path())?;
        let primary = Arc::new(PacedEmbedder {
            id: "http://bench#bench-primary".into(),
            model: "bench-primary",
            dim: PRIMARY_DIM,
            per_call: call,
            per_text: text,
        });
        let secondary: Option<Arc<dyn Embedder>> = two_spaces.then(|| {
            Arc::new(PacedEmbedder {
                id: "http://bench#bench-secondary".into(),
                model: "bench-secondary",
                dim: SECONDARY_DIM,
                per_call: call,
                per_text: text,
            }) as Arc<dyn Embedder>
        });
        let embedder = Arc::new(CompetitiveEmbedder::new(primary, secondary));
        let engine = RetrievalEngine::new(store.clone(), dir.path(), embedder.clone())?;
        let mut spaces = vec![embedder.primary_space().to_string()];
        if let Some(s) = embedder.secondary_space() {
            spaces.push(s.to_string());
        }

        let start = Instant::now();
        for i in 0..count {
            let content = format!("benchmark memory {i}: a short note about topic {}", i % 37);
            let mut experience = Experience {
                content: content.clone(),
                ..Default::default()
            };
            let (vector, secondary) = embedder.encode_dual(&content)?;
            let space = embedder.primary_space_for(&vector);
            experience.set_primary_vector(vector, space);
            if let Some(v) = secondary {
                if let Some(space) = embedder.secondary_space_for(&v) {
                    experience.set_secondary_vector(v, space);
                }
            }
            let memory = Memory::new(
                MemoryId(uuid::Uuid::new_v4()),
                experience,
                0.5,
                None,
                None,
                None,
                None,
            );
            store.store(&memory)?;
            engine.index_memory(&memory)?;
        }
        let elapsed = start.elapsed();
        rows.push(Row::Write {
            mode: "inline",
            spaces,
            memories: count,
            per_memory_us: (elapsed.as_micros() / count.max(1) as u128) as u64,
            embed_call_us: call.as_micros() as u64,
            embed_text_us: text.as_micros() as u64,
        });
    }
    Ok(rows)
}

fn measure_drain(texts: &[String]) -> Vec<Row> {
    let Ok(url) = std::env::var("VELD_BENCH_EMBEDDER_URL") else {
        return vec![Row::Skipped {
            what: "drain".into(),
            why: "VELD_BENCH_EMBEDDER_URL is not set".into(),
        }];
    };
    let model = std::env::var("VELD_BENCH_EMBEDDER_MODEL")
        .unwrap_or_else(|_| "text-embedding-nomic-embed-text-v1.5".into());
    let embedder = HttpEmbedder::new(HttpEmbedderConfig::from_id(&format!("{url}#{model}")));
    if let Err(e) = embedder.probe() {
        return vec![Row::Skipped {
            what: format!("drain against {url}"),
            why: format!("probe failed: {e:#}"),
        }];
    }
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let mut rows = Vec::new();
    for batch in DRAIN_BATCHES {
        let start = Instant::now();
        let mut done = 0usize;
        let mut failure = None;
        for chunk in refs.chunks(batch) {
            match embedder.encode_batch(chunk) {
                Ok(v) => done += v.len(),
                Err(e) => {
                    failure = Some(format!("{e:#}"));
                    break;
                }
            }
        }
        match failure {
            Some(why) => rows.push(Row::Skipped {
                what: format!("drain batch {batch} against {url}"),
                why,
            }),
            None => rows.push(Row::Drain {
                host: url.clone(),
                model: model.clone(),
                batch_size: batch,
                texts: done,
                texts_per_sec: done as f64 / start.elapsed().as_secs_f64().max(1e-9),
            }),
        }
    }
    rows
}

fn git_sha() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn report_path(generated_at: &str) -> PathBuf {
    std::env::var("VELD_BENCH_SPACE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("evaluations/spaces")
                .join(format!("{}.json", generated_at.replace(':', "")))
        })
}

fn print_table(report: &Report) {
    for row in &report.rows {
        match row {
            Row::Search {
                memories,
                spaces,
                codec,
                p50_us,
                p95_us,
                p99_us,
                ..
            } => println!(
                "search    {memories:>7}  {:<48} {codec:<4} p50 {p50_us:>6}us  p95 {p95_us:>6}us  p99 {p99_us:>6}us",
                spaces.join(" + ")
            ),
            Row::Query {
                memories,
                mode,
                p50_us,
                p95_us,
                p99_us,
                p95_over_one_space,
                ..
            } => println!(
                "query     {memories:>7}  {mode:<53} p50 {p50_us:>6}us  p95 {p95_us:>6}us  p99 {p99_us:>6}us  p95 {p95_over_one_space:.2}x one space"
            ),
            Row::Footprint {
                memories,
                space,
                codec,
                bytes,
                rss_growth_bytes,
                build_ms,
            } => println!(
                "footprint {memories:>7}  {space:<48} {codec:<4} {:>8} KiB stored  rss +{:>8} KiB  build {build_ms} ms",
                bytes / 1024,
                rss_growth_bytes.map_or_else(|| "?".to_string(), |b| (b / 1024).to_string())
            ),
            Row::TwoSpaces {
                memories,
                one_space_bytes,
                two_spaces_bytes,
                bytes_ratio,
                rss_ratio,
            } => println!(
                "two/one   {memories:>7}  {:>8} KiB vs {:>8} KiB stored: {bytes_ratio:.2}x  rss {}",
                two_spaces_bytes / 1024,
                one_space_bytes / 1024,
                rss_ratio.map_or_else(|| "?".to_string(), |r| format!("{r:.2}x"))
            ),
            Row::Quality {
                memories,
                space,
                codec,
                recall_at_10,
                ..
            } => println!("quality   {memories:>7}  {space:<48} {codec:<4} recall@10 {recall_at_10:.3}"),
            Row::Write {
                mode,
                spaces,
                memories,
                per_memory_us,
                ..
            } => println!(
                "write     {memories:>7}  {:<48} {mode:<6} {per_memory_us} us/memory",
                spaces.join(" + ")
            ),
            Row::Drain {
                host,
                batch_size,
                texts_per_sec,
                ..
            } => println!("drain     {host} batch {batch_size:>3}: {texts_per_sec:.1} texts/s"),
            Row::Skipped { what, why } => println!("skipped   {what}: {why}"),
        }
    }
}

fn main() -> Result<()> {
    // `cargo bench` passes `--bench`; a filter argument is not meaningful for this harness.
    let sizes = env_sizes();
    let queries = env_usize("VELD_BENCH_SPACE_QUERIES", 200);
    let writes = env_usize("VELD_BENCH_WRITE_COUNT", 300);
    let call = Duration::from_micros(env_usize("VELD_BENCH_EMBED_CALL_US", 3000) as u64);
    let text = Duration::from_micros(env_usize("VELD_BENCH_EMBED_TEXT_US", 200) as u64);

    let mut rows = measure_search_and_footprint(&sizes, queries, call + text)?;
    rows.extend(measure_inline_writes(writes, call, text)?);
    let texts: Vec<String> = (0..DRAIN_TEXTS)
        .map(|i| format!("drain text {i}: the embedder sees a paragraph of ordinary length, {i}"))
        .collect();
    rows.extend(measure_drain(&texts));

    let generated_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let report = Report {
        generated_at: generated_at.clone(),
        git_sha: git_sha(),
        sizes,
        rows,
    };
    print_table(&report);
    let path = report_path(&generated_at);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("writing {}", path.display()))?;
    println!("report: {}", path.display());
    Ok(())
}
