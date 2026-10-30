//! Measurements for `VELD_MODEL_IDLE_UNLOAD_SECS`, ignored by default: they need the cached
//! MiniLM, BERT-NER and cross-encoder files and run for minutes. Run one per process, because
//! the setting is read once:
//!
//! ```text
//! VELD_MODEL_IDLE_UNLOAD_SECS=120 <test binary> --ignored --exact idle_rss_profile --nocapture
//! VELD_MODEL_IDLE_UNLOAD_SECS=4   <test binary> --ignored --exact first_query_after_idle --nocapture
//! ```
//!
//! `VELD_IDLE_PROBE_MINUTES` (default `0,5,30`) sets when `idle_rss_profile` samples RSS;
//! `VELD_IDLE_PROBE_CYCLES` (default 30) sets how many idle-then-query cycles the other runs.

use std::time::{Duration, Instant};

use veld::embeddings::cross_encoder::{default_cross_encoder, CrossEncoderModel};
use veld::embeddings::minilm::{EmbeddingConfig, MiniLMEmbedder};
use veld::embeddings::ner::{NerConfig, NeuralNer};
use veld::embeddings::Embedder;

struct Models {
    embedder: MiniLMEmbedder,
    reranker: veld::embeddings::cross_encoder::OnnxCrossEncoder,
    ner: NeuralNer,
}

impl Models {
    fn load() -> Self {
        let models = Self {
            embedder: MiniLMEmbedder::new(EmbeddingConfig::default()).expect("MiniLM"),
            reranker: default_cross_encoder(),
            ner: NeuralNer::new(NerConfig::default()).expect("NER"),
        };
        assert!(!models.embedder.is_simplified(), "MiniLM files not cached");
        assert!(!models.ner.is_fallback_mode(), "BERT-NER files not cached");
        models
    }

    /// One call to each model, on text no cache has seen; the latency of each.
    fn query(&self, round: usize) -> [Duration; 3] {
        let text = format!("Round {round}: Alice met Bob in Paris to review the Veld release.");
        let docs = [
            format!("Round {round} notes from the Paris review."),
            format!("Round {round}: an unrelated grocery list."),
        ];
        let doc_refs: Vec<&str> = docs.iter().map(String::as_str).collect();

        let t = Instant::now();
        self.embedder.encode(&text).expect("embed");
        let embed = t.elapsed();
        let t = Instant::now();
        self.reranker.score_pairs(&text, &doc_refs).expect("rerank");
        let rerank = t.elapsed();
        let t = Instant::now();
        self.ner.extract(&text).expect("ner");
        [embed, rerank, t.elapsed()]
    }
}

fn rss_mb() -> f64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<f64>()
        .expect("rss in KiB")
        / 1024.0
}

/// macOS compresses an idle process's pages and RSS stops counting them, so an idle process
/// with every model loaded still shows a small RSS. The physical footprint counts compressed
/// memory, so it is what tells a loaded model from a dropped one there.
fn footprint_mb() -> Option<f64> {
    let out = std::process::Command::new("footprint")
        .args(["-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let rest = text.split("Footprint:").nth(1)?;
    let mut words = rest.split_whitespace();
    let value: f64 = words.next()?.parse().ok()?;
    let scale = match words.next()? {
        "B" => 1.0 / (1024.0 * 1024.0),
        "KB" => 1.0 / 1024.0,
        "MB" => 1.0,
        "GB" => 1024.0,
        _ => return None,
    };
    Some(value * scale)
}

fn memory() -> String {
    match footprint_mb() {
        Some(footprint) => format!("rss_mb={:.1} footprint_mb={footprint:.1}", rss_mb()),
        None => format!("rss_mb={:.1}", rss_mb()),
    }
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[test]
#[ignore = "measurement: needs cached models and runs for up to 30 minutes"]
fn idle_rss_profile() {
    let unload = env_or("VELD_MODEL_IDLE_UNLOAD_SECS", "off");
    let before = memory();
    let models = Models::load();
    models.query(0);
    let start = Instant::now();
    println!("idle_unload={unload} before_load {before}");
    for minute in env_or("VELD_IDLE_PROBE_MINUTES", "0,5,30").split(',') {
        let minute: u64 = minute.trim().parse().expect("minutes");
        let at = Duration::from_secs(minute * 60);
        if let Some(wait) = at.checked_sub(start.elapsed()) {
            std::thread::sleep(wait);
        }
        println!(
            "idle_unload={unload} idle_min={minute} {} minilm_loaded={}",
            memory(),
            models.embedder.is_model_loaded()
        );
    }
}

#[test]
#[ignore = "measurement: needs cached models and VELD_MODEL_IDLE_UNLOAD_SECS"]
fn first_query_after_idle() {
    let unload = env_or("VELD_MODEL_IDLE_UNLOAD_SECS", "off");
    let pause = match unload.parse::<u64>() {
        Ok(secs) if secs > 0 => Duration::from_secs(secs + 3),
        _ => Duration::from_secs(5),
    };
    let cycles: usize = env_or("VELD_IDLE_PROBE_CYCLES", "30")
        .parse()
        .expect("cycles");
    let models = Models::load();
    models.query(0);

    let mut samples: [Vec<f64>; 3] = Default::default();
    let mut totals = Vec::with_capacity(cycles);
    for round in 1..=cycles {
        std::thread::sleep(pause);
        let reloaded = !models.embedder.is_model_loaded();
        let took = models.query(round);
        let mut total = 0.0;
        for (i, d) in took.iter().enumerate() {
            let ms = d.as_secs_f64() * 1000.0;
            samples[i].push(ms);
            total += ms;
        }
        totals.push(total);
        println!(
            "round={round} reloaded={reloaded} embed_ms={:.1} rerank_ms={:.1} ner_ms={:.1}",
            samples[0][round - 1],
            samples[1][round - 1],
            samples[2][round - 1]
        );
    }
    for (name, values) in ["embed", "rerank", "ner", "total"]
        .iter()
        .zip(samples.iter_mut().chain(std::iter::once(&mut totals)))
    {
        values.sort_by(f64::total_cmp);
        println!(
            "idle_unload={unload} {name} p50_ms={:.1} p99_ms={:.1} max_ms={:.1}",
            percentile(values, 50.0),
            percentile(values, 99.0),
            values[values.len() - 1]
        );
    }
}
