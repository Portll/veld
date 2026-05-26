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
        let header_bytes =
            bincode::serde::encode_to_vec(alignment.header(), bincode::config::standard())
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
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).context("reading magic")?;
    if &magic != MAGIC {
        bail!(
            "alignment file magic mismatch at {}: expected {:?}, got {:?}",
            path.display(),
            MAGIC,
            magic
        );
    }

    let mut header_len_bytes = [0u8; 4];
    r.read_exact(&mut header_len_bytes)?;
    let header_len = u32::from_le_bytes(header_len_bytes) as usize;
    let mut header_buf = vec![0u8; header_len];
    r.read_exact(&mut header_buf)?;
    let (header, _consumed): (AlignmentHeader, usize) =
        bincode::serde::decode_from_slice(&header_buf, bincode::config::standard())
            .context("decoding alignment header")?;

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
                self.dim,
                secondary.len()
            );
        }
        Ok(secondary.to_vec())
    }
    fn in_dim(&self) -> usize {
        self.dim
    }
    fn out_dim(&self) -> usize {
        self.dim
    }
    fn pair_id(&self) -> &AlignmentPairId {
        &self.pair_id
    }
    fn method(&self) -> &'static str {
        "identity"
    }
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
    fn payload_bytes(&self) -> Vec<u8> {
        Vec::new()
    }
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
        if path.exists() {
            return Some(path);
        }
    }
    if let Some(home) = dirs::home_dir() {
        let p = home
            .join(".cache")
            .join("veld")
            .join("alignments")
            .join(pair_id.cache_filename());
        if p.exists() {
            return Some(p);
        }
    }
    let bundled = std::path::PathBuf::from("assets/alignments").join(pair_id.cache_filename());
    if bundled.exists() {
        return Some(bundled);
    }
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
