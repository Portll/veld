# Embedding Alignment — Implementation Plan

**Companion to:** `EMBEDDING_ALIGNMENT.md`
**Author:** john@portll.net
**Date:** 2026-05-27
**Target branch:** `feat/alignment-core` (and later `feat/alignment-mcp-experimental`)

This document is a copy-paste-into-VS-Code implementation plan. Every code block is meant to be lifted whole into the file path named above it. No TODOs, no stubs, no placeholders — per `CLAUDE.md`.

Work runs in seven phases. Each phase ends with a verification command. Run `cargo check` / `cargo clippy` between phases; the user runs `cargo build` and trunk separately.

---

## 0. Prerequisites

Check that veld's `Cargo.toml` already has these (it does, but confirm):

- `anyhow`
- `serde` (with `derive`)
- `serde_json`
- `bincode`
- `tracing`
- `clap` (with `derive`)
- `rayon` (used elsewhere; we'll use it for parallel encoding)

We will add one new dependency: `nalgebra` for SVD and matrix solves.

---

## 1. Branch and dependency setup

```bash
# from C:\Repositories\Portll\veld
git checkout -b feat/alignment-core
```

Edit `Cargo.toml`. Under `[dependencies]`, add (or confirm) the line:

```toml
nalgebra = { version = "0.33", default-features = false, features = ["std", "matrixmultiply"] }
```

Under `[[bin]]` entries, add three new binaries:

```toml
[[bin]]
name = "alignment-collect"
path = "src/bin/alignment_collect.rs"

[[bin]]
name = "alignment-fit"
path = "src/bin/alignment_fit.rs"

[[bin]]
name = "alignment-eval"
path = "src/bin/alignment_eval.rs"
```

Verification:

```bash
cargo check -p veld
```

---

## 2. Phase 1 — Scaffolding

Three new files: the alignment trait + identity impl + persistence framework; the module wire-up; and a unit test set.

### 2.1 — `src/embeddings/alignment.rs` (new)

```rust
//! Embedding alignment abstraction.
//!
//! Provides a learned projection from a secondary embedder's latent space
//! into the primary embedder's latent space. Cross-embedder vector math is
//! only permitted *through* an alignment — see `CompetitiveEmbedder` for the
//! enforcing wrapper.
//!
//! # File format (magic `VELDALN1`)
//!
//! ```text
//! [ 8 bytes ] magic = "VELDALN1"
//! [ 4 bytes ] header_len  (u32 LE)
//! [ N bytes ] header      (bincode-encoded AlignmentHeader)
//! [ 8 bytes ] payload_len (u64 LE)
//! [ M bytes ] payload     (method-specific)
//! ```

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"VELDALN1";

/// Identifier pair for the embedders an alignment was fitted on.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AlignmentPairId {
    pub primary: String,
    pub secondary: String,
}

impl AlignmentPairId {
    pub fn new(primary: impl Into<String>, secondary: impl Into<String>) -> Self {
        Self {
            primary: primary.into(),
            secondary: secondary.into(),
        }
    }

    /// Filename used in `~/.cache/veld/alignments/`.
    pub fn cache_filename(&self) -> String {
        // Sanitize — model names can contain slashes (e.g. nomic-ai/nomic-embed-text-v1.5).
        let p = self.primary.replace('/', "_");
        let s = self.secondary.replace('/', "_");
        format!("{p}__{s}.bin")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentHeader {
    pub method: String,
    pub pair_id: AlignmentPairId,
    pub in_dim: usize,
    pub out_dim: usize,
    pub fit_unix_ts: i64,
    pub eval_paired_cosine_mean: Option<f32>,
}

pub trait Alignment: Send + Sync {
    fn project(&self, secondary: &[f32]) -> Result<Vec<f32>>;

    fn project_batch(&self, batch: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        batch.iter().map(|v| self.project(v)).collect()
    }

    fn in_dim(&self) -> usize;
    fn out_dim(&self) -> usize;
    fn pair_id(&self) -> &AlignmentPairId;
    fn method(&self) -> &'static str;
    fn header(&self) -> AlignmentHeader;
    fn payload_bytes(&self) -> Vec<u8>;
}

/// Current unix timestamp, seconds. Falls back to 0 on system-clock errors.
pub fn unix_ts_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Persist any `Alignment` implementor to disk atomically.
pub fn save_alignment(path: &Path, alignment: &dyn Alignment) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating alignment dir {}", parent.display()))?;
    }
    let tmp = path.with_extension("bin.tmp");
    {
        let file = File::create(&tmp)
            .with_context(|| format!("opening {} for write", tmp.display()))?;
        let mut w = BufWriter::new(file);

        w.write_all(MAGIC)?;
        let header_bytes = bincode::serialize(&alignment.header())
            .context("serializing alignment header")?;
        w.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
        w.write_all(&header_bytes)?;

        let payload = alignment.payload_bytes();
        w.write_all(&(payload.len() as u64).to_le_bytes())?;
        w.write_all(&payload)?;
        w.flush()?;
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Read header and payload from disk for dispatch by method.
pub fn read_alignment_file(path: &Path) -> Result<(AlignmentHeader, Vec<u8>)> {
    let file = File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).context("reading magic")?;
    if &magic != MAGIC {
        bail!(
            "alignment file magic mismatch at {}: expected {:?}, got {:?}",
            path.display(), MAGIC, magic
        );
    }

    let mut header_len_bytes = [0u8; 4];
    r.read_exact(&mut header_len_bytes)?;
    let header_len = u32::from_le_bytes(header_len_bytes) as usize;
    let mut header_buf = vec![0u8; header_len];
    r.read_exact(&mut header_buf)?;
    let header: AlignmentHeader =
        bincode::deserialize(&header_buf).context("decoding alignment header")?;

    let mut payload_len_bytes = [0u8; 8];
    r.read_exact(&mut payload_len_bytes)?;
    let payload_len = u64::from_le_bytes(payload_len_bytes) as usize;
    let mut payload = vec![0u8; payload_len];
    r.read_exact(&mut payload)?;

    Ok((header, payload))
}

// -----------------------------------------------------------------------------
// IdentityAlignment
// -----------------------------------------------------------------------------

/// Identity projection — used when both embedders are the same model.
pub struct IdentityAlignment {
    dim: usize,
    pair_id: AlignmentPairId,
}

impl IdentityAlignment {
    pub fn new(dim: usize, pair_id: AlignmentPairId) -> Self {
        Self { dim, pair_id }
    }
}

impl Alignment for IdentityAlignment {
    fn project(&self, secondary: &[f32]) -> Result<Vec<f32>> {
        if secondary.len() != self.dim {
            bail!(
                "identity alignment dim mismatch: expected {}, got {}",
                self.dim, secondary.len()
            );
        }
        Ok(secondary.to_vec())
    }
    fn in_dim(&self) -> usize { self.dim }
    fn out_dim(&self) -> usize { self.dim }
    fn pair_id(&self) -> &AlignmentPairId { &self.pair_id }
    fn method(&self) -> &'static str { "identity" }
    fn header(&self) -> AlignmentHeader {
        AlignmentHeader {
            method: self.method().to_string(),
            pair_id: self.pair_id.clone(),
            in_dim: self.dim,
            out_dim: self.dim,
            fit_unix_ts: unix_ts_now(),
            eval_paired_cosine_mean: Some(1.0),
        }
    }
    fn payload_bytes(&self) -> Vec<u8> { Vec::new() }
}

// -----------------------------------------------------------------------------
// Resolution: where the global alignment lives.
// -----------------------------------------------------------------------------

/// Resolve the path of the global alignment for a given pair, by precedence:
///
/// 1. `$VELD_ALIGNMENT_PATH` (operator override, exact file path).
/// 2. `~/.cache/veld/alignments/<primary>__<secondary>.bin`.
/// 3. `<repo>/assets/alignments/<primary>__<secondary>.bin` (bundled default).
///
/// Returns `None` if no candidate file exists; callers may then trigger fit.
pub fn resolve_alignment_path(pair_id: &AlignmentPairId) -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("VELD_ALIGNMENT_PATH") {
        let path = std::path::PathBuf::from(p);
        if path.exists() { return Some(path); }
    }
    if let Some(home) = dirs::home_dir() {
        let p = home
            .join(".cache").join("veld").join("alignments")
            .join(pair_id.cache_filename());
        if p.exists() { return Some(p); }
    }
    let bundled = std::path::PathBuf::from("assets/alignments")
        .join(pair_id.cache_filename());
    if bundled.exists() { return Some(bundled); }
    None
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn pid() -> AlignmentPairId {
        AlignmentPairId::new("nomic-embed-text-v1.5", "minilm-l6-v2")
    }

    #[test]
    fn identity_projects_unchanged() {
        let a = IdentityAlignment::new(4, pid());
        let v = vec![0.5_f32, -0.5, 0.5, -0.5];
        let p = a.project(&v).unwrap();
        assert_eq!(p, v);
    }

    #[test]
    fn identity_rejects_wrong_dim() {
        let a = IdentityAlignment::new(4, pid());
        assert!(a.project(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn save_load_round_trip_identity() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("id.bin");
        let a = IdentityAlignment::new(8, pid());
        save_alignment(&path, &a).unwrap();
        let (header, payload) = read_alignment_file(&path).unwrap();
        assert_eq!(header.method, "identity");
        assert_eq!(header.in_dim, 8);
        assert_eq!(header.out_dim, 8);
        assert_eq!(header.pair_id, pid());
        assert!(payload.is_empty());
    }

    #[test]
    fn bad_magic_is_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.bin");
        std::fs::write(&path, b"NOTVELD!........").unwrap();
        assert!(read_alignment_file(&path).is_err());
    }

    #[test]
    fn pair_id_cache_filename_sanitizes_slashes() {
        let id = AlignmentPairId::new("nomic-ai/nomic-embed-text-v1.5", "minilm-l6-v2");
        let f = id.cache_filename();
        assert!(!f.contains('/'));
        assert!(f.ends_with(".bin"));
    }
}
```

### 2.2 — Add `dirs` and `tempfile` to `Cargo.toml`

If not already present:

```toml
[dependencies]
dirs = "5"

[dev-dependencies]
tempfile = "3"
```

### 2.3 — `src/embeddings/mod.rs` (patch)

Add the new module and re-exports. Locate the `pub mod` block and add:

```rust
pub mod alignment;
pub mod alignment_procrustes;
pub mod alignment_ridge;
```

Locate the re-exports section and add:

```rust
pub use alignment::{
    read_alignment_file, resolve_alignment_path, save_alignment, unix_ts_now,
    Alignment, AlignmentHeader, AlignmentPairId, IdentityAlignment,
};
pub use alignment_procrustes::ProcrustesAlignment;
pub use alignment_ridge::RidgeAlignment;
```

(`alignment_procrustes` and `alignment_ridge` are created in phases 3 and 5; the `pub mod` declarations will produce a compile error until those files exist. Defer the `pub mod` lines until those phases if you want clean intermediate builds.)

### 2.4 — Verification

```bash
cargo check -p veld
cargo test -p veld embeddings::alignment::tests
```

All five tests should pass.

---

## 3. Phase 2 — Corpus assembly

We build a pairs file: text rows plus encoded vectors from both embedders. The format is two files that share an index:

- `pairs.jsonl` — one JSON object per line, fields `{text, text_hash, domain, source, license}`.
- `pairs.vec` — packed binary, one row per pairs.jsonl line: `[primary_vec (d_p × f32 LE)][secondary_vec (d_s × f32 LE)]`. A 16-byte header `(d_p:u32 LE, d_s:u32 LE, n:u64 LE)` precedes the rows.

### 3.1 — Directory layout (create empty)

```bash
mkdir -p evaluations/alignment/{pairs,fitted,prototypes}
```

### 3.2 — `src/bin/alignment_collect.rs` (new)

This binary builds `pairs.jsonl` from declared sources. It does not encode vectors — encoding is the job of `alignment-fit`, since it depends on the loaded embedders.

```rust
//! Corpus-assembly CLI.
//!
//! Ingests text from declared sources (filesystem trees, friends-contributed
//! JSONL files), de-duplicates by content hash, applies per-domain quotas, and
//! writes a `pairs.jsonl` ready to be encoded by `alignment-fit`.

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "alignment-collect")]
struct Args {
    /// Output file (.jsonl)
    #[arg(short, long, default_value = "evaluations/alignment/pairs/pairs.jsonl")]
    out: PathBuf,

    /// Per-domain target row count (rows past the quota are dropped).
    #[arg(long, default_value_t = 5000)]
    quota: usize,

    /// Source spec files (TOML). Multiple allowed.
    #[arg(long)]
    sources: Vec<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct SourceSpec {
    /// e.g. "web_development", "programming", "ai_loop"
    domain: String,
    /// "filesystem" | "jsonl"
    kind: String,
    /// path or directory
    path: PathBuf,
    /// for filesystem: file extensions to include (e.g. ["rs","md","sql"])
    #[serde(default)]
    extensions: Vec<String>,
    /// per-row license tag
    license: String,
    /// source label written to output (e.g. "veld-repo", "stackoverflow")
    source: String,
    /// for filesystem: minimum chunk size in bytes
    #[serde(default = "default_min")]
    min_chunk: usize,
    /// for filesystem: maximum chunk size in bytes
    #[serde(default = "default_max")]
    max_chunk: usize,
}

fn default_min() -> usize { 64 }
fn default_max() -> usize { 2048 }

#[derive(Debug, Serialize)]
struct PairRow {
    text: String,
    text_hash: String,
    domain: String,
    source: String,
    license: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.sources.is_empty() {
        bail!("at least one --sources spec file is required");
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut per_domain: HashMap<String, usize> = HashMap::new();

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let out_file = OpenOptions::new()
        .create(true).write(true).truncate(true)
        .open(&args.out)
        .with_context(|| format!("opening {}", args.out.display()))?;
    let mut out_writer = BufWriter::new(out_file);

    for spec_path in &args.sources {
        let spec_bytes = std::fs::read_to_string(spec_path)
            .with_context(|| format!("reading {}", spec_path.display()))?;
        let spec: SourceSpec = toml::from_str(&spec_bytes)
            .with_context(|| format!("parsing {}", spec_path.display()))?;

        eprintln!(
            "[collect] domain={} kind={} path={}",
            spec.domain, spec.kind, spec.path.display()
        );

        match spec.kind.as_str() {
            "filesystem" => {
                collect_filesystem(&spec, &mut seen, &mut per_domain, args.quota, &mut out_writer)?;
            }
            "jsonl" => {
                collect_jsonl(&spec, &mut seen, &mut per_domain, args.quota, &mut out_writer)?;
            }
            other => bail!("unknown source kind: {other}"),
        }
    }

    out_writer.flush()?;
    eprintln!("[collect] per-domain row counts:");
    for (d, n) in &per_domain {
        eprintln!("  {d}: {n}");
    }
    Ok(())
}

fn hash_text(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    format!("{:x}", h.finalize())
}

fn write_row<W: Write>(
    w: &mut W,
    row: PairRow,
    seen: &mut HashSet<String>,
    per_domain: &mut HashMap<String, usize>,
    quota: usize,
) -> Result<bool> {
    if !seen.insert(row.text_hash.clone()) { return Ok(false); }
    let count = per_domain.entry(row.domain.clone()).or_insert(0);
    if *count >= quota { return Ok(false); }
    *count += 1;
    serde_json::to_writer(&mut *w, &row)?;
    w.write_all(b"\n")?;
    Ok(true)
}

fn collect_filesystem<W: Write>(
    spec: &SourceSpec,
    seen: &mut HashSet<String>,
    per_domain: &mut HashMap<String, usize>,
    quota: usize,
    w: &mut W,
) -> Result<()> {
    let exts: HashSet<&str> = spec.extensions.iter().map(String::as_str).collect();
    for entry in WalkDir::new(&spec.path).follow_links(false).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() { continue; }
        if !exts.is_empty() {
            let Some(ext) = entry.path().extension().and_then(|s| s.to_str()) else { continue };
            if !exts.contains(ext) { continue; }
        }
        let Ok(content) = std::fs::read_to_string(entry.path()) else { continue };
        for chunk in chunk_text(&content, spec.min_chunk, spec.max_chunk) {
            let row = PairRow {
                text_hash: hash_text(&chunk),
                text: chunk,
                domain: spec.domain.clone(),
                source: spec.source.clone(),
                license: spec.license.clone(),
            };
            let written = write_row(w, row, seen, per_domain, quota)?;
            if !written && per_domain.get(&spec.domain).copied().unwrap_or(0) >= quota {
                return Ok(());
            }
        }
    }
    Ok(())
}

fn collect_jsonl<W: Write>(
    spec: &SourceSpec,
    seen: &mut HashSet<String>,
    per_domain: &mut HashMap<String, usize>,
    quota: usize,
    w: &mut W,
) -> Result<()> {
    #[derive(Deserialize)]
    struct Incoming { text: String }
    let file = File::open(&spec.path)
        .with_context(|| format!("opening {}", spec.path.display()))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let incoming: Incoming = serde_json::from_str(&line)?;
        if incoming.text.len() < spec.min_chunk { continue; }
        if incoming.text.len() > spec.max_chunk { continue; }
        let row = PairRow {
            text_hash: hash_text(&incoming.text),
            text: incoming.text,
            domain: spec.domain.clone(),
            source: spec.source.clone(),
            license: spec.license.clone(),
        };
        let _ = write_row(w, row, seen, per_domain, quota)?;
        if per_domain.get(&spec.domain).copied().unwrap_or(0) >= quota { break; }
    }
    Ok(())
}

/// Greedy paragraph-respecting chunker. Splits on blank lines first; if a
/// paragraph exceeds `max`, breaks on the nearest line boundary.
fn chunk_text(content: &str, min: usize, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for para in content.split("\n\n") {
        let para = para.trim();
        if para.is_empty() { continue; }
        if buf.len() + para.len() + 2 > max {
            if buf.len() >= min { out.push(std::mem::take(&mut buf)); }
            if para.len() > max {
                for line in para.lines() {
                    if buf.len() + line.len() + 1 > max {
                        if buf.len() >= min { out.push(std::mem::take(&mut buf)); }
                    }
                    if !buf.is_empty() { buf.push('\n'); }
                    buf.push_str(line);
                }
            } else {
                buf.push_str(para);
            }
        } else {
            if !buf.is_empty() { buf.push_str("\n\n"); }
            buf.push_str(para);
        }
    }
    if buf.len() >= min { out.push(buf); }
    out
}
```

Add `walkdir`, `toml`, and `sha2` to `Cargo.toml` if not present:

```toml
walkdir = "2"
toml = "0.8"
sha2 = "0.10"
```

### 3.3 — Source specs

Create `evaluations/alignment/pairs/sources/` and one TOML per source. Example for the veld repo itself:

```toml
# evaluations/alignment/pairs/sources/veld-rust.toml
domain = "programming"
kind = "filesystem"
path = "."
extensions = ["rs"]
license = "AGPL-3.0"
source = "veld-repo-rust"
min_chunk = 256
max_chunk = 2048
```

Repeat per domain. The full set we ship lives in section 9.

### 3.4 — Verification

```bash
cargo run --release --bin alignment-collect -- \
  --sources evaluations/alignment/pairs/sources/veld-rust.toml \
  --out evaluations/alignment/pairs/pairs.jsonl \
  --quota 1000

wc -l evaluations/alignment/pairs/pairs.jsonl
```

Expect a non-empty file with at most 1000 lines for the `programming` domain.

---

## 4. Phase 3 — Procrustes fit

Two files: the Procrustes alignment itself, and the fit CLI.

### 4.1 — `src/embeddings/alignment_procrustes.rs` (new)

```rust
//! Orthogonal Procrustes alignment.
//!
//! Solves R* = argmin_R ||A - B R||_F  subject to R columns orthonormal,
//! via SVD of M = B^T A. Closed-form, no hyperparameters.

use anyhow::{anyhow, bail, Result};
use nalgebra::{DMatrix, DVector};

use super::alignment::{
    unix_ts_now, Alignment, AlignmentHeader, AlignmentPairId,
};

pub struct ProcrustesAlignment {
    pair_id: AlignmentPairId,
    in_dim: usize,
    out_dim: usize,
    /// Shape (out_dim × in_dim) — project = rotation * secondary_vec.
    rotation: DMatrix<f32>,
    fit_unix_ts: i64,
    eval_paired_cosine_mean: Option<f32>,
}

impl ProcrustesAlignment {
    /// `primary[i]` and `secondary[i]` must be paired (same source text).
    /// Both sides must be L2-normalized (norm within [0.99, 1.01]).
    pub fn fit(
        pair_id: AlignmentPairId,
        primary: &[Vec<f32>],
        secondary: &[Vec<f32>],
    ) -> Result<Self> {
        if primary.len() != secondary.len() {
            bail!(
                "primary/secondary length mismatch: {} vs {}",
                primary.len(), secondary.len()
            );
        }
        if primary.is_empty() {
            bail!("cannot fit Procrustes on empty pairs");
        }
        let n = primary.len();
        let d_p = primary[0].len();
        let d_s = secondary[0].len();
        if d_p == 0 || d_s == 0 {
            bail!("zero-dimension embeddings: d_p={d_p}, d_s={d_s}");
        }
        for (i, v) in primary.iter().enumerate() {
            if v.len() != d_p { bail!("primary row {i}: dim {} != {d_p}", v.len()); }
            assert_normalized(v, "primary", i)?;
        }
        for (i, v) in secondary.iter().enumerate() {
            if v.len() != d_s { bail!("secondary row {i}: dim {} != {d_s}", v.len()); }
            assert_normalized(v, "secondary", i)?;
        }

        // A: n × d_p, B: n × d_s
        let a = DMatrix::<f32>::from_row_iterator(n, d_p, primary.iter().flatten().copied());
        let b = DMatrix::<f32>::from_row_iterator(n, d_s, secondary.iter().flatten().copied());

        // M = B^T A → (d_s × d_p)
        let m = b.transpose() * &a;

        let svd = m.svd(true, true);
        let u   = svd.u.ok_or_else(|| anyhow!("SVD U missing"))?;       // d_s × k
        let v_t = svd.v_t.ok_or_else(|| anyhow!("SVD V^T missing"))?;   // k × d_p

        let q = u * v_t;                  // d_s × d_p, optimal projection
        let rotation = q.transpose();     // d_p × d_s, for project = rotation · s

        Ok(Self {
            pair_id,
            in_dim: d_s,
            out_dim: d_p,
            rotation,
            fit_unix_ts: unix_ts_now(),
            eval_paired_cosine_mean: None,
        })
    }

    pub fn set_eval(&mut self, paired_cosine_mean: f32) {
        self.eval_paired_cosine_mean = Some(paired_cosine_mean);
    }

    /// Reconstruct from persisted payload bytes (row-major f32 LE).
    pub fn from_payload(header: AlignmentHeader, payload: &[u8]) -> Result<Self> {
        let expected = header.in_dim * header.out_dim * 4;
        if payload.len() != expected {
            bail!(
                "procrustes payload size mismatch: expected {} bytes ({}×{}×4), got {}",
                expected, header.out_dim, header.in_dim, payload.len()
            );
        }
        let floats: Vec<f32> = payload.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let rotation = DMatrix::<f32>::from_row_slice(header.out_dim, header.in_dim, &floats);
        Ok(Self {
            pair_id: header.pair_id,
            in_dim: header.in_dim,
            out_dim: header.out_dim,
            rotation,
            fit_unix_ts: header.fit_unix_ts,
            eval_paired_cosine_mean: header.eval_paired_cosine_mean,
        })
    }
}

fn assert_normalized(v: &[f32], side: &str, idx: usize) -> Result<()> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if (norm - 1.0).abs() > 0.01 {
        bail!("{} vector {} not L2-normalized (norm={:.4})", side, idx, norm);
    }
    Ok(())
}

impl Alignment for ProcrustesAlignment {
    fn project(&self, secondary: &[f32]) -> Result<Vec<f32>> {
        if secondary.len() != self.in_dim {
            bail!(
                "procrustes project: expected in_dim={}, got {}",
                self.in_dim, secondary.len()
            );
        }
        let x = DVector::<f32>::from_row_slice(secondary);
        let y = &self.rotation * x;
        Ok(y.iter().copied().collect())
    }

    fn in_dim(&self) -> usize { self.in_dim }
    fn out_dim(&self) -> usize { self.out_dim }
    fn pair_id(&self) -> &AlignmentPairId { &self.pair_id }
    fn method(&self) -> &'static str { "orthogonal_procrustes" }

    fn header(&self) -> AlignmentHeader {
        AlignmentHeader {
            method: self.method().to_string(),
            pair_id: self.pair_id.clone(),
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            fit_unix_ts: self.fit_unix_ts,
            eval_paired_cosine_mean: self.eval_paired_cosine_mean,
        }
    }

    fn payload_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.in_dim * self.out_dim * 4);
        for r in 0..self.out_dim {
            for c in 0..self.in_dim {
                out.extend_from_slice(&self.rotation[(r, c)].to_le_bytes());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::alignment::{read_alignment_file, save_alignment};
    use tempfile::tempdir;

    fn pid() -> AlignmentPairId {
        AlignmentPairId::new("modelA-768", "modelB-768")
    }

    fn norm(v: &mut [f32]) {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 { for x in v.iter_mut() { *x /= n; } }
    }

    /// When secondary == rotation(primary), Procrustes should recover the rotation.
    #[test]
    fn fits_identity_when_pairs_match() {
        let mut rng = vec![
            vec![0.4_f32, 0.3, 0.7, 0.5],
            vec![-0.2,    0.9, 0.1, 0.3],
            vec![0.5,    -0.4, 0.6, 0.5],
            vec![0.1,     0.1, 0.9, 0.4],
            vec![0.7,    -0.3, 0.5, 0.4],
        ];
        for v in rng.iter_mut() { norm(v); }
        let primary = rng.clone();
        let secondary = rng.clone();

        let a = ProcrustesAlignment::fit(pid(), &primary, &secondary).unwrap();
        let projected = a.project(&secondary[0]).unwrap();
        // identity-ish; cosine should be near 1
        let cos: f32 = projected.iter().zip(primary[0].iter()).map(|(x,y)| x*y).sum();
        assert!(cos > 0.99, "expected near-1 cosine, got {cos}");
    }

    #[test]
    fn save_load_round_trip() {
        let mut rows: Vec<Vec<f32>> = (0..20).map(|i| {
            let mut v = vec![0.0_f32; 8];
            for (j, x) in v.iter_mut().enumerate() {
                *x = ((i * 7 + j) as f32 * 0.13).sin();
            }
            norm(&mut v);
            v
        }).collect();
        let primary = rows.clone();
        for v in rows.iter_mut() { v[0] += 0.05; norm(v); }
        let secondary = rows;
        let a = ProcrustesAlignment::fit(pid(), &primary, &secondary).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("p.bin");
        save_alignment(&path, &a).unwrap();
        let (header, payload) = read_alignment_file(&path).unwrap();
        let b = ProcrustesAlignment::from_payload(header, &payload).unwrap();

        let pa = a.project(&secondary[0]).unwrap();
        let pb = b.project(&secondary[0]).unwrap();
        for (x, y) in pa.iter().zip(pb.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_unnormalized_input() {
        let primary = vec![vec![10.0_f32, 0.0, 0.0]];
        let secondary = vec![vec![1.0_f32, 0.0, 0.0]];
        assert!(ProcrustesAlignment::fit(pid(), &primary, &secondary).is_err());
    }
}
```

### 4.2 — `src/bin/alignment_fit.rs` (new)

This encodes `pairs.jsonl` with both embedders, splits 90/10, fits Procrustes, evaluates on the held-out 10%, and persists to `evaluations/alignment/fitted/<pair_id>.bin`.

```rust
//! Fit a global alignment from a pairs file.

use anyhow::{bail, Context, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

use veld::embeddings::{
    save_alignment, Alignment, AlignmentPairId, Embedder, ProcrustesAlignment,
};

#[derive(Parser)]
#[command(name = "alignment-fit")]
struct Args {
    #[arg(long, default_value = "evaluations/alignment/pairs/pairs.jsonl")]
    pairs: PathBuf,

    #[arg(long, default_value = "evaluations/alignment/fitted")]
    out_dir: PathBuf,

    /// Held-out fraction.
    #[arg(long, default_value_t = 0.1)]
    holdout: f32,

    /// Acceptance threshold on mean paired cosine for the held-out split.
    #[arg(long, default_value_t = 0.80)]
    min_cosine: f32,

    /// Primary model identifier (free-form, used in pair_id).
    #[arg(long)]
    primary_id: String,

    /// Secondary model identifier.
    #[arg(long)]
    secondary_id: String,
}

#[derive(Debug, Deserialize)]
struct PairRow { text: String, domain: String }

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt::init();

    let pair_id = AlignmentPairId::new(args.primary_id.clone(), args.secondary_id.clone());

    // Load both embedders from veld's standard factory paths.
    let primary: Arc<dyn Embedder> = veld::embeddings::load_primary_embedder(&args.primary_id)
        .context("loading primary embedder")?;
    let secondary: Arc<dyn Embedder> = veld::embeddings::load_secondary_embedder(&args.secondary_id)
        .context("loading secondary embedder")?;

    // Stream pairs.
    let file = File::open(&args.pairs)
        .with_context(|| format!("opening {}", args.pairs.display()))?;
    let rows: Vec<PairRow> = BufReader::new(file).lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();
    if rows.is_empty() { bail!("no rows in {}", args.pairs.display()); }
    tracing::info!("loaded {} text rows", rows.len());

    // Encode in parallel chunks (Rayon).
    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    let primary_vecs: Vec<Vec<f32>> = texts.par_iter()
        .map(|t| primary.encode(t))
        .collect::<Result<Vec<_>>>()?;
    let secondary_vecs: Vec<Vec<f32>> = texts.par_iter()
        .map(|t| secondary.encode(t))
        .collect::<Result<Vec<_>>>()?;

    // Deterministic 90/10 split by text_hash mod 10 (so re-runs are stable).
    let mut train_p = Vec::new();
    let mut train_s = Vec::new();
    let mut hold_p  = Vec::new();
    let mut hold_s  = Vec::new();
    let cutoff = (args.holdout * 10.0).round() as usize;
    for (i, row) in rows.iter().enumerate() {
        let bucket = stable_bucket(&row.text);
        if bucket < cutoff {
            hold_p.push(primary_vecs[i].clone());
            hold_s.push(secondary_vecs[i].clone());
        } else {
            train_p.push(primary_vecs[i].clone());
            train_s.push(secondary_vecs[i].clone());
        }
    }
    tracing::info!("train={} holdout={}", train_p.len(), hold_p.len());

    // Fit.
    let mut alignment = ProcrustesAlignment::fit(pair_id.clone(), &train_p, &train_s)
        .context("Procrustes fit")?;

    // Evaluate.
    let cos = mean_paired_cosine(&alignment, &hold_p, &hold_s)?;
    tracing::info!("held-out mean paired cosine: {:.4}", cos);
    alignment.set_eval(cos);

    if cos < args.min_cosine {
        bail!(
            "alignment quality {:.4} below acceptance threshold {:.4} — refusing to install",
            cos, args.min_cosine
        );
    }

    std::fs::create_dir_all(&args.out_dir)?;
    let out_path = args.out_dir.join(pair_id.cache_filename());
    save_alignment(&out_path, &alignment)?;
    tracing::info!("installed alignment at {}", out_path.display());
    Ok(())
}

fn mean_paired_cosine(
    a: &ProcrustesAlignment, primary: &[Vec<f32>], secondary: &[Vec<f32>],
) -> Result<f32> {
    if primary.is_empty() { return Ok(0.0); }
    let mut sum = 0.0_f32;
    for (p, s) in primary.iter().zip(secondary.iter()) {
        let projected = a.project(s)?;
        let dot: f32 = projected.iter().zip(p.iter()).map(|(x,y)| x*y).sum();
        let np: f32 = projected.iter().map(|x| x*x).sum::<f32>().sqrt();
        let pp: f32 = p.iter().map(|x| x*x).sum::<f32>().sqrt();
        sum += dot / (np * pp).max(1e-12);
    }
    Ok(sum / primary.len() as f32)
}

fn stable_bucket(text: &str) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    (h.finish() as usize) % 10
}
```

Note the two helper imports `load_primary_embedder` / `load_secondary_embedder` — add these to `src/embeddings/mod.rs` as thin wrappers around the existing factory paths (`nomic.rs`, `minilm.rs`, `http_embedder.rs`). Use the same path resolution used by the server.

### 4.3 — `src/bin/alignment_eval.rs` (new)

```rust
//! Evaluate an installed alignment against a held-out pairs file.

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

use veld::embeddings::{
    read_alignment_file, AlignmentPairId, Embedder, ProcrustesAlignment,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    alignment: PathBuf,
    #[arg(long)]
    pairs: PathBuf,
    #[arg(long)]
    primary_id: String,
    #[arg(long)]
    secondary_id: String,
}

#[derive(Deserialize)]
struct Row { text: String, domain: String }

fn main() -> Result<()> {
    let args = Args::parse();
    let (header, payload) = read_alignment_file(&args.alignment)?;
    let expected_pid = AlignmentPairId::new(&args.primary_id, &args.secondary_id);
    if header.pair_id != expected_pid {
        anyhow::bail!(
            "pair_id mismatch: file = {:?}, expected = {:?}",
            header.pair_id, expected_pid
        );
    }
    let alignment = ProcrustesAlignment::from_payload(header, &payload)?;

    let primary: Arc<dyn Embedder> = veld::embeddings::load_primary_embedder(&args.primary_id)?;
    let secondary: Arc<dyn Embedder> = veld::embeddings::load_secondary_embedder(&args.secondary_id)?;

    let file = File::open(&args.pairs)
        .with_context(|| format!("opening {}", args.pairs.display()))?;
    let rows: Vec<Row> = BufReader::new(file).lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();

    let mut by_domain: std::collections::BTreeMap<String, (f32, usize)> = Default::default();
    let mut overall: (f32, usize) = (0.0, 0);

    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    let pv: Vec<Vec<f32>> = texts.par_iter().map(|t| primary.encode(t)).collect::<Result<_>>()?;
    let sv: Vec<Vec<f32>> = texts.par_iter().map(|t| secondary.encode(t)).collect::<Result<_>>()?;

    for (i, row) in rows.iter().enumerate() {
        let projected = alignment.project(&sv[i])?;
        let dot: f32 = projected.iter().zip(pv[i].iter()).map(|(x,y)| x*y).sum();
        let np: f32 = projected.iter().map(|x| x*x).sum::<f32>().sqrt();
        let pp: f32 = pv[i].iter().map(|x| x*x).sum::<f32>().sqrt();
        let cos = dot / (np * pp).max(1e-12);
        let e = by_domain.entry(row.domain.clone()).or_insert((0.0, 0));
        e.0 += cos; e.1 += 1;
        overall.0 += cos; overall.1 += 1;
    }

    println!("=== alignment eval ===");
    println!("file: {}", args.alignment.display());
    println!("method: {}", alignment.method());
    println!("overall mean paired cosine: {:.4} (n={})",
        overall.0 / overall.1.max(1) as f32, overall.1);
    println!("per-domain:");
    for (d, (sum, n)) in &by_domain {
        println!("  {d}: {:.4} (n={n})", sum / *n as f32);
    }
    Ok(())
}
```

### 4.4 — Verification

```bash
cargo test -p veld embeddings::alignment_procrustes::tests
cargo run --release --bin alignment-fit -- \
  --primary-id nomic-embed-text-v1.5 \
  --secondary-id minilm-l6-v2 \
  --pairs evaluations/alignment/pairs/pairs.jsonl
cargo run --release --bin alignment-eval -- \
  --alignment evaluations/alignment/fitted/nomic-embed-text-v1.5__minilm-l6-v2.bin \
  --pairs evaluations/alignment/pairs/pairs.jsonl \
  --primary-id nomic-embed-text-v1.5 \
  --secondary-id minilm-l6-v2
```

Tests pass; the fit reports a held-out cosine that should be ≥ 0.80.

---

## 5. Phase 4 — Retrieval integration

Three patches: `CompetitiveEmbedder` gets an alignment field and `encode_aligned`; the retrieval engine gets `search_aligned`; the fused-position writer lands the regular task space.

### 5.1 — Patch `src/embeddings/competitive.rs`

Add an import at the top:

```rust
use super::alignment::Alignment;
```

Extend the struct:

```rust
pub struct CompetitiveEmbedder {
    primary: Arc<dyn Embedder>,
    secondary: Option<Arc<dyn Embedder>>,
    secondary_dim: Option<usize>,
    alignment: Option<Arc<dyn Alignment>>,
}
```

Update both constructors to pass `alignment: None`:

```rust
pub fn new(primary: Arc<dyn Embedder>, secondary: Option<Arc<dyn Embedder>>) -> Self {
    let secondary_dim = secondary.as_ref().map(|e| e.dimension());
    Self { primary, secondary, secondary_dim, alignment: None }
}

pub fn primary_only(primary: Arc<dyn Embedder>) -> Self {
    Self { primary, secondary: None, secondary_dim: None, alignment: None }
}
```

Add a builder for installing an alignment:

```rust
/// Install a learned alignment. Refuses if dimensions don't match the
/// currently-configured secondary, or if no secondary is configured.
pub fn with_alignment(mut self, alignment: Arc<dyn Alignment>) -> anyhow::Result<Self> {
    use anyhow::bail;
    let sec_dim = self.secondary_dim.ok_or_else(||
        anyhow::anyhow!("cannot install alignment: no secondary embedder configured")
    )?;
    if alignment.in_dim() != sec_dim {
        bail!(
            "alignment in_dim {} != secondary dim {}",
            alignment.in_dim(), sec_dim
        );
    }
    if alignment.out_dim() != self.primary.dimension() {
        bail!(
            "alignment out_dim {} != primary dim {}",
            alignment.out_dim(), self.primary.dimension()
        );
    }
    self.alignment = Some(alignment);
    Ok(self)
}

/// True if an alignment is installed.
pub fn has_alignment(&self) -> bool { self.alignment.is_some() }

/// Encode with the secondary, then project to primary space via the alignment.
/// Returns `Ok(None)` if either the secondary or the alignment is absent.
pub fn encode_aligned(&self, text: &str) -> anyhow::Result<Option<Vec<f32>>> {
    let (Some(secondary), Some(alignment)) = (&self.secondary, &self.alignment) else {
        return Ok(None);
    };
    let s = secondary.encode(text)?;
    Ok(Some(alignment.project(&s)?))
}

/// Encode dual and additionally produce the *fused primary-space vector* used
/// for the "regular task space" — a weighted blend of primary and projected
/// secondary. `alpha` is the weight on the projected secondary; primary gets
/// `1 - alpha`. Returns `None` for the fused vector when the alignment or
/// secondary is absent. The result is L2-renormalized.
pub fn encode_fused(&self, text: &str, alpha: f32) -> anyhow::Result<(Vec<f32>, Option<Vec<f32>>)> {
    let (primary_emb, secondary_emb) = self.encode_dual(text)?;
    let fused = match (secondary_emb, &self.alignment) {
        (Some(s), Some(a)) => {
            let projected = a.project(&s)?;
            let mut out: Vec<f32> = primary_emb.iter().zip(projected.iter())
                .map(|(p, q)| (1.0 - alpha) * p + alpha * q)
                .collect();
            let norm: f32 = out.iter().map(|x| x*x).sum::<f32>().sqrt();
            if norm > 0.0 { for x in &mut out { *x /= norm; } }
            Some(out)
        }
        _ => None,
    };
    Ok((primary_emb, fused))
}
```

### 5.2 — Patch `src/memory/retrieval.rs`

Add a new public method on the retrieval engine. Locate the impl block alongside `search_ids_secondary` and add:

```rust
/// Search the *primary* Vamana index using a query encoded with the
/// secondary embedder and projected through the installed alignment.
///
/// Returns an empty vec if no alignment is installed or no secondary
/// embedder is configured.
pub fn search_ids_aligned(
    &self,
    query_text: &str,
    limit: usize,
) -> anyhow::Result<Vec<(crate::memory::MemoryId, f32)>> {
    let Some(projected) = self.embedder.encode_aligned(query_text)? else {
        return Ok(Vec::new());
    };
    // Reuse the primary Vamana search path.
    let primary_index = self.vector_index.read();
    let id_map = self.id_mapping.read();
    let candidates = primary_index.search(
        &projected,
        limit * crate::constants::VECTOR_SEARCH_CANDIDATE_MULTIPLIER * 2,
    )?;
    let mut out = Vec::with_capacity(candidates.len().min(limit));
    for (vid, score) in candidates.into_iter().take(limit) {
        if let Some(mid) = id_map.vector_to_memory(vid) {
            out.push((mid, score));
        }
    }
    Ok(out)
}
```

(Field names `vector_index` and `id_mapping` follow the existing primary names in `retrieval.rs`; adjust to actual identifiers if they differ.)

### 5.3 — Prototype set

Create `evaluations/alignment/prototypes/README.md`:

```markdown
# Task-space prototypes

Each prototype is a JSON file: a named landmark in the primary-space
"regular task space." Prototypes are edited by hand and recomputed
(field `centroid_vec`) on demand from the `example_ids`.

## Schema

```json
{
  "name": "react_component",
  "domain": "web_development",
  "description": "A standalone React functional component, typically TSX.",
  "example_ids": ["mem_abc", "mem_def", "mem_ghi"],
  "centroid_vec": [/* d_primary floats; recomputed by `alignment-fit prototypes` */]
}
```

## Inventory (target ~30–50)

- web_development: react_component, vue_component, rest_endpoint, graphql_resolver,
  css_module, html_fragment, form_validation, frontend_bug_report
- programming: docstring, commit_message, code_review_comment, refactor_note,
  unit_test, integration_test, build_config, stack_trace_summary
- project_management: ticket, milestone, retrospective_note, status_update,
  rfc_abstract, design_doc_summary, release_note
- database: ddl_schema, migration, query_plan, index_strategy, er_caption
- analytics: sql_query, dashboard_spec, metric_definition, ab_test_writeup
- devops: pipeline_config, runbook, incident_report, slo_statement
- docs: readme_section, adr, api_reference, changelog_entry
- security: threat_model, dependency_audit, code_review_finding
- testing: test_plan, fixture_definition, flake_report
- ai_loop: prompt_definition, tool_schema, eval_entry, model_output_critique
```

(Populating these is data work — out of scope for code.)

### 5.4 — Verification

```bash
cargo check -p veld
cargo clippy -p veld -- -D warnings
```

---

## 6. Phase 5 — Ridge fallback

### 6.1 — `src/embeddings/alignment_ridge.rs` (new)

```rust
//! Ridge-regression alignment — unconstrained linear projection with L2
//! regularisation. Used when Procrustes residuals are too large (the two
//! spaces are not isometric).

use anyhow::{anyhow, bail, Result};
use nalgebra::{DMatrix, DVector};

use super::alignment::{
    unix_ts_now, Alignment, AlignmentHeader, AlignmentPairId,
};

pub struct RidgeAlignment {
    pair_id: AlignmentPairId,
    in_dim: usize,
    out_dim: usize,
    weight: DMatrix<f32>, // (out_dim × in_dim)
    lambda: f32,
    fit_unix_ts: i64,
    eval_paired_cosine_mean: Option<f32>,
}

impl RidgeAlignment {
    /// Solve W = (B^T B + λI)^{-1} B^T A. W shape: (d_s × d_p), stored
    /// transposed for project-by-multiply.
    pub fn fit(
        pair_id: AlignmentPairId,
        primary: &[Vec<f32>],
        secondary: &[Vec<f32>],
        lambda: f32,
    ) -> Result<Self> {
        if primary.len() != secondary.len() {
            bail!("primary/secondary length mismatch");
        }
        if primary.is_empty() { bail!("empty fit"); }
        if lambda < 0.0 { bail!("lambda must be non-negative"); }
        let n = primary.len();
        let d_p = primary[0].len();
        let d_s = secondary[0].len();

        let a = DMatrix::<f32>::from_row_iterator(n, d_p, primary.iter().flatten().copied());
        let b = DMatrix::<f32>::from_row_iterator(n, d_s, secondary.iter().flatten().copied());

        let btb = b.transpose() * &b;
        let reg = DMatrix::<f32>::identity(d_s, d_s) * lambda;
        let lhs = btb + reg;
        let rhs = b.transpose() * &a; // (d_s × d_p)
        let inv = lhs.try_inverse()
            .ok_or_else(|| anyhow!("B^T B + λI not invertible — try larger λ"))?;
        let w = inv * rhs; // (d_s × d_p)
        let weight = w.transpose(); // (d_p × d_s)

        Ok(Self {
            pair_id, in_dim: d_s, out_dim: d_p, weight, lambda,
            fit_unix_ts: unix_ts_now(),
            eval_paired_cosine_mean: None,
        })
    }

    pub fn set_eval(&mut self, paired_cosine_mean: f32) {
        self.eval_paired_cosine_mean = Some(paired_cosine_mean);
    }

    /// Payload layout: `[lambda: f32 LE][weight: (out_dim × in_dim) f32 LE row-major]`.
    pub fn from_payload(header: AlignmentHeader, payload: &[u8]) -> Result<Self> {
        if payload.len() < 4 { bail!("ridge payload too short"); }
        let lambda = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        let mat = &payload[4..];
        let expected = header.in_dim * header.out_dim * 4;
        if mat.len() != expected {
            bail!("ridge payload size mismatch: expected {} got {}", expected, mat.len());
        }
        let floats: Vec<f32> = mat.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let weight = DMatrix::<f32>::from_row_slice(header.out_dim, header.in_dim, &floats);
        Ok(Self {
            pair_id: header.pair_id,
            in_dim: header.in_dim,
            out_dim: header.out_dim,
            weight,
            lambda,
            fit_unix_ts: header.fit_unix_ts,
            eval_paired_cosine_mean: header.eval_paired_cosine_mean,
        })
    }
}

impl Alignment for RidgeAlignment {
    fn project(&self, secondary: &[f32]) -> Result<Vec<f32>> {
        if secondary.len() != self.in_dim {
            bail!("ridge project: expected in_dim={}, got {}", self.in_dim, secondary.len());
        }
        let x = DVector::<f32>::from_row_slice(secondary);
        let y = &self.weight * x;
        Ok(y.iter().copied().collect())
    }
    fn in_dim(&self) -> usize { self.in_dim }
    fn out_dim(&self) -> usize { self.out_dim }
    fn pair_id(&self) -> &AlignmentPairId { &self.pair_id }
    fn method(&self) -> &'static str { "ridge" }
    fn header(&self) -> AlignmentHeader {
        AlignmentHeader {
            method: self.method().to_string(),
            pair_id: self.pair_id.clone(),
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            fit_unix_ts: self.fit_unix_ts,
            eval_paired_cosine_mean: self.eval_paired_cosine_mean,
        }
    }
    fn payload_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.in_dim * self.out_dim * 4);
        out.extend_from_slice(&self.lambda.to_le_bytes());
        for r in 0..self.out_dim {
            for c in 0..self.in_dim {
                out.extend_from_slice(&self.weight[(r, c)].to_le_bytes());
            }
        }
        out
    }
}
```

### 6.2 — Wire Ridge into `alignment-fit`

Add a CLI flag `--method procrustes|ridge` (default `procrustes`). When `ridge` is selected, fit `RidgeAlignment::fit(pair_id, &train_p, &train_s, 1e-2)` (lambda configurable via `--lambda`). Acceptance check is the same.

---

## 7. Phase 6 — Embedder-onboarding hook

Auto-fit on first startup with a new embedder pair.

### 7.1 — Patch the retrieval engine constructor

In `src/memory/retrieval.rs`, in the `new_with_storage_path` (or equivalent) constructor, after the embedder and secondary index have been initialised, add:

```rust
// Onboarding hook: if an alignment exists for this pair, load it. If none
// exists and a pairs file is present, fit one. Otherwise proceed without.
if let Some(sec_dim) = embedder.secondary_dimension() {
    let pair_id = crate::embeddings::AlignmentPairId::new(
        embedder.primary_embedder().model_id(),
        embedder.secondary_embedder().expect("checked").model_id(),
    );
    let aligned = crate::memory::alignment_onboarding::resolve_or_fit(
        &pair_id, sec_dim, embedder.primary_dimension(),
    )?;
    if let Some(a) = aligned {
        embedder = std::sync::Arc::new(
            std::sync::Arc::try_unwrap(embedder)
                .map_err(|_| anyhow::anyhow!("embedder shared during onboarding"))?
                .with_alignment(a)?
        );
    }
}
```

(`Embedder::model_id()` is a new trait method — default `"unknown"`, overridden in each concrete embedder to return its canonical identifier. Add it to the `Embedder` trait in `mod.rs` with a default.)

### 7.2 — `src/memory/alignment_onboarding.rs` (new)

```rust
//! Embedder-onboarding hook for alignment.

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;

use crate::embeddings::{
    read_alignment_file, resolve_alignment_path, save_alignment, Alignment,
    AlignmentPairId, ProcrustesAlignment,
};

/// Returns an alignment for the given pair, fitting if necessary.
///
/// Order of resolution:
/// 1. Existing file resolved via `resolve_alignment_path`.
/// 2. Auto-fit from the bundled pairs file if `VELD_ALIGNMENT_AUTOFIT=1`.
/// 3. `Ok(None)` — caller falls back to max-score union (no regression).
pub fn resolve_or_fit(
    pair_id: &AlignmentPairId,
    _sec_dim: usize,
    _pri_dim: usize,
) -> Result<Option<Arc<dyn Alignment>>> {
    if let Some(path) = resolve_alignment_path(pair_id) {
        let (header, payload) = read_alignment_file(&path)?;
        if header.pair_id != *pair_id {
            tracing::warn!(
                "alignment file pair_id {:?} != current {:?}; ignoring",
                header.pair_id, pair_id
            );
            return Ok(None);
        }
        let alignment = ProcrustesAlignment::from_payload(header, &payload)?;
        tracing::info!("loaded alignment from {}", path.display());
        return Ok(Some(Arc::new(alignment)));
    }

    if std::env::var("VELD_ALIGNMENT_AUTOFIT").as_deref() != Ok("1") {
        tracing::info!(
            "no alignment for {:?}; set VELD_ALIGNMENT_AUTOFIT=1 to auto-fit",
            pair_id
        );
        return Ok(None);
    }

    let pairs_path = std::env::var("VELD_ALIGNMENT_PAIRS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("evaluations/alignment/pairs/pairs.jsonl"));
    if !pairs_path.exists() {
        tracing::warn!("auto-fit requested but pairs file missing: {}", pairs_path.display());
        return Ok(None);
    }

    tracing::info!("auto-fitting alignment for {:?}", pair_id);
    let alignment = fit_from_pairs_file(&pairs_path, pair_id)?;
    let cache_dir = dirs::home_dir()
        .map(|h| h.join(".cache").join("veld").join("alignments"))
        .unwrap_or_else(|| PathBuf::from("/tmp/veld/alignments"));
    let out_path = cache_dir.join(pair_id.cache_filename());
    save_alignment(&out_path, &alignment)?;
    tracing::info!("installed alignment at {}", out_path.display());
    Ok(Some(Arc::new(alignment)))
}

fn fit_from_pairs_file(
    pairs_path: &std::path::Path,
    pair_id: &AlignmentPairId,
) -> Result<ProcrustesAlignment> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use rayon::prelude::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Row { text: String }

    let file = File::open(pairs_path)?;
    let rows: Vec<Row> = BufReader::new(file).lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();
    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();

    let primary = crate::embeddings::load_primary_embedder(&pair_id.primary)?;
    let secondary = crate::embeddings::load_secondary_embedder(&pair_id.secondary)?;
    let pv: Vec<Vec<f32>> = texts.par_iter().map(|t| primary.encode(t)).collect::<Result<_>>()?;
    let sv: Vec<Vec<f32>> = texts.par_iter().map(|t| secondary.encode(t)).collect::<Result<_>>()?;

    let mut alignment = ProcrustesAlignment::fit(pair_id.clone(), &pv, &sv)?;
    // Self-cosine on training data as a sanity gate.
    let n_check = pv.len().min(200);
    let mut sum = 0.0_f32;
    for i in 0..n_check {
        let proj = alignment.project(&sv[i])?;
        let dot: f32 = proj.iter().zip(pv[i].iter()).map(|(x,y)| x*y).sum();
        sum += dot;
    }
    alignment.set_eval(sum / n_check as f32);
    Ok(alignment)
}
```

Register the module in `src/memory/mod.rs`:

```rust
pub mod alignment_onboarding;
```

### 7.3 — Verification

```bash
VELD_ALIGNMENT_AUTOFIT=1 cargo run --release --bin veld-server
# inspect log: should show "loaded alignment from ..." OR "auto-fitting alignment for ..."
```

---

## 8. Phase 7 — MCP experimental branch

After phases 1–6 land on `feat/alignment-core`, cut the experimental MCP surface on a separate branch:

```bash
git checkout feat/alignment-core
git checkout -b feat/alignment-mcp-experimental
```

### 8.1 — Add MCP tool definitions

In `mcp-server/src/tools/`, add a new file (TypeScript) that exposes:

- `recall_aligned(query: string, limit?: number)` — uses `search_ids_aligned` on the Rust side via the existing veld HTTP API.
- `find_related_projects(query: string, limit?: number)` — uses the prototype set: nearest prototype + nearest aligned-space neighbours.

Tool descriptions MUST start with `[experimental]` and reference the feature flag.

### 8.2 — Feature flag

In `mcp-server/src/config.ts`, add:

```ts
export const ALIGNMENT_EXPERIMENTAL_ENABLED =
  process.env.VELD_MCP_ALIGNMENT_EXPERIMENTAL === "1";
```

In the tool registration code, skip registering `recall_aligned` / `find_related_projects` unless the flag is on.

### 8.3 — Promotion criteria

Promotion from experimental to stable requires two consecutive quarterly evaluations on the frozen test set in `evaluations/alignment/` to clear the per-domain cosine ≥ 0.85 and retrieval recall@10 within 5pp of the same-space baseline. Document this in `evaluations/alignment/PROMOTION_CRITERIA.md`.

---

## 9. Sources spec inventory

Create these files under `evaluations/alignment/pairs/sources/`:

| File | Domain | Source |
|---|---|---|
| `veld-rust.toml` | programming | filesystem, ext `rs` |
| `veld-md.toml` | docs | filesystem, ext `md` |
| `sleight-rust.toml` | programming | filesystem, ext `rs`, path `../sleight` |
| `web-public.toml` | web_development | jsonl from public scrape |
| `pm-public.toml` | project_management | jsonl |
| `db-public.toml` | database | jsonl |
| `analytics-public.toml` | analytics | jsonl |
| `devops-public.toml` | devops | jsonl |
| `security-public.toml` | security | jsonl |
| `testing-public.toml` | testing | jsonl |
| `ai-loop-public.toml` | ai_loop | jsonl |
| `friends/*.toml` | various | jsonl, one per contributor |

For each public-source JSONL, ship a `fetch.sh` that downloads the source corpus to a gitignored location, applies a license filter, and writes `<domain>-public.jsonl`. The fetch scripts are stored in `evaluations/alignment/pairs/sources/fetch/`.

---

## 10. Contribution guide

Create `evaluations/alignment/CONTRIBUTING.md`:

```markdown
# Contributing alignment training data

Thanks for offering pairs. The alignment training corpus is **code and public
sources only** — no email, no chat threads, no document stores, no personal
correspondence. Every row must be redistributable under a permissive licence.

## Submission format

One JSONL file, UTF-8, one object per line:

```json
{ "text": "<the text>", "domain": "<domain>", "license": "<SPDX id>", "source": "<your label>" }
```

Domains: `web_development | project_management | database | programming |
analytics | devops | docs | security | testing | ai_loop`.

License: SPDX identifier (`MIT`, `Apache-2.0`, `BSD-3-Clause`, `CC-BY-SA-4.0`,
`CC0-1.0`, etc.). Rows under any other licence are rejected.

## Size and shape

- 64–2048 characters per row.
- 500–5000 rows per contribution is a good sweet spot.
- Diverse within a domain (don't submit 5000 rows from a single file).
- De-duplicated on your end.

## Process

1. Open a PR adding `evaluations/alignment/pairs/sources/friends/<your-handle>.toml`
   plus the JSONL file (gitignored by default — confirm the licence permits
   redistribution before committing the data itself).
2. The CI alignment-fit job picks up the new source and re-fits the global
   alignment.
3. The evaluation harness reports the cosine and recall deltas. If quality
   improves, the PR is merged.

## Code of conduct

Don't submit anything you wouldn't want indexed by a public search engine.
```

---

## 11. Verification matrix

| Phase | Command | Pass criterion |
|---|---|---|
| 1 | `cargo test embeddings::alignment::tests` | 5/5 pass |
| 2 | `cargo run --bin alignment-collect -- ...` | non-empty pairs.jsonl, no panic |
| 3 | `cargo test embeddings::alignment_procrustes::tests` | 3/3 pass |
| 3 | `cargo run --bin alignment-fit -- ...` | held-out cosine ≥ 0.80 |
| 3 | `cargo run --bin alignment-eval -- ...` | per-domain table prints |
| 4 | `cargo clippy -- -D warnings` | no warnings |
| 5 | `cargo test embeddings::alignment_ridge` | added tests pass |
| 6 | `VELD_ALIGNMENT_AUTOFIT=1 cargo run --bin veld-server` | startup log shows load or fit |
| 7 | `VELD_MCP_ALIGNMENT_EXPERIMENTAL=1` MCP probe | `recall_aligned` registered |

---

## 12. Commit sequence

One commit per phase. Suggested messages:

```
feat(embeddings): scaffold Alignment trait and IdentityAlignment
feat(embeddings): alignment-collect CLI for pairs.jsonl assembly
feat(embeddings): Procrustes alignment + alignment-fit/eval binaries
feat(retrieval): wire alignment into CompetitiveEmbedder and search_ids_aligned
feat(embeddings): Ridge alignment fallback
feat(retrieval): embedder-onboarding hook for auto-fit
feat(mcp): experimental recall_aligned and find_related_projects tools
docs(alignment): contribution guide and prototype README
```

No "Generated with Claude Code" or Co-Authored-By lines — per `CLAUDE.md`.

---

## 13. Rollback

Each phase is additive. To revert any phase without affecting earlier work:

- Phase 7 (MCP): unset `VELD_MCP_ALIGNMENT_EXPERIMENTAL`, drop the branch.
- Phase 6 (hook): unset `VELD_ALIGNMENT_AUTOFIT`, remove the constructor block; existing alignment files are still loaded passively.
- Phase 4 (retrieval integration): delete the alignment file from `~/.cache/veld/alignments/`; `encode_aligned` returns `Ok(None)` and `search_ids_aligned` returns an empty vec — no regression.
- Phases 1–3 and 5 are pure additions; reverting is a `git revert <hash>`.

The no-alignment install path is byte-identical to today at every phase.
