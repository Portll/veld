//! One-off: seal the journal frames written before the journal was encrypted.
//!
//! Until the journal took its key from the store keystore, a tenant sealed by
//! `VELD_MASTER_PASSPHRASE` alone wrote its intent log in plaintext, and the log is
//! append-only and never compacted, so those frames stay on disk as plaintext
//! memory snapshots. This rewrites a log with every payload sealed exactly as
//! [`super::IntentLog::append`] seals one: same key, same (empty) associated data,
//! same LSN, CRC recomputed over the sealed bytes. Frames already sealed are copied
//! after they are proved to open under the key, so a wrong key fails before anything
//! is written.
//!
//! Run it with the server stopped. The rewrite goes to a temporary file, is read back
//! through the key and compared frame for frame with what was read, and replaces the
//! log by rename only if the log did not change while it ran. A log with a corrupt
//! tail is refused: truncate it first, so nothing past the damage is silently dropped.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};

use super::{probe_mode, IntentLogIter};
use crate::keystore::SecretCryptor;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ResealReport {
    pub path: PathBuf,
    pub frames: u64,
    /// Plaintext frames sealed by this run (or that a dry run would seal).
    pub resealed: u64,
    pub already_sealed: u64,
    /// Whether the log on disk was replaced.
    pub rewritten: bool,
}

/// Fold one frame into the run's digest: its LSN and its plaintext payload.
fn fold(digest: &mut Sha256, lsn: u64, plaintext: &[u8]) {
    digest.update(lsn.to_le_bytes());
    digest.update((plaintext.len() as u64).to_le_bytes());
    digest.update(plaintext);
}

fn write_frame(out: &mut impl Write, lsn: u64, sealed: &[u8]) -> std::io::Result<()> {
    let len = sealed.len() as u32;
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&len.to_le_bytes());
    hasher.update(&lsn.to_le_bytes());
    hasher.update(sealed);
    out.write_all(&len.to_le_bytes())?;
    out.write_all(&lsn.to_le_bytes())?;
    out.write_all(sealed)?;
    out.write_all(&hasher.finalize().to_le_bytes())
}

pub fn reseal_plaintext_frames(
    path: &Path,
    cryptor: &SecretCryptor,
    dry_run: bool,
) -> Result<ResealReport> {
    let before = std::fs::metadata(path).with_context(|| format!("{path:?}"))?;
    let frame_start = probe_mode(path)?.frame_region_start();
    let tmp = path.with_extension("intentlog.reseal.tmp");

    let mut report = ResealReport {
        path: path.to_path_buf(),
        frames: 0,
        resealed: 0,
        already_sealed: 0,
        rewritten: false,
    };
    let mut read_digest = Sha256::new();
    let mut out = if dry_run {
        None
    } else {
        let mut header = vec![0u8; frame_start as usize];
        File::open(path)?.read_exact(&mut header)?;
        let mut w = BufWriter::new(File::create(&tmp).with_context(|| format!("{tmp:?}"))?);
        w.write_all(&header)?;
        Some(w)
    };

    let outcome = (|| -> Result<()> {
        let mut frames = IntentLogIter::open_for_scan(path)?;
        while let Some(record) = frames.next() {
            let record = record.map_err(|e| {
                anyhow!(
                    "frame unreadable at offset {} ({e}); truncate the corrupt tail before resealing",
                    frames.current_offset()
                )
            })?;
            report.frames += 1;
            let (plaintext, sealed) = if SecretCryptor::is_encrypted(&record.payload) {
                report.already_sealed += 1;
                let plaintext = cryptor
                    .decrypt_bytes(&record.payload, b"")
                    .with_context(|| {
                        format!(
                            "frame at LSN {} is sealed under a different key; refusing to mix keys in one log",
                            record.lsn.0
                        )
                    })?;
                (plaintext, record.payload)
            } else {
                report.resealed += 1;
                let sealed = cryptor.encrypt_bytes(&record.payload, b"")?;
                (record.payload, sealed)
            };
            fold(&mut read_digest, record.lsn.0, &plaintext);
            if let Some(w) = out.as_mut() {
                write_frame(w, record.lsn.0, &sealed)?;
            }
        }
        Ok(())
    })();

    let Some(mut w) = out else {
        return outcome.map(|()| report);
    };
    if let Err(e) = outcome {
        drop(w);
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    w.flush()?;
    w.get_ref().sync_data()?;
    drop(w);

    if report.resealed == 0 {
        std::fs::remove_file(&tmp)?;
        return Ok(report);
    }

    // Read the rewrite back through the key: every frame sealed, every LSN and
    // payload the same as what was read.
    let mut written_digest = Sha256::new();
    let mut written = 0u64;
    let mut frames = IntentLogIter::open_for_scan(&tmp)?;
    while let Some(record) = frames.next() {
        let record = record.context("the rewritten log does not read back")?;
        if !SecretCryptor::is_encrypted(&record.payload) {
            bail!(
                "the rewritten log still holds a plaintext frame at LSN {}",
                record.lsn.0
            );
        }
        fold(
            &mut written_digest,
            record.lsn.0,
            &cryptor.decrypt_bytes(&record.payload, b"")?,
        );
        written += 1;
    }
    if written != report.frames || written_digest.finalize() != read_digest.finalize() {
        let _ = std::fs::remove_file(&tmp);
        bail!("the rewritten log does not match the original frame for frame; left unchanged");
    }

    let now = std::fs::metadata(path)?;
    if now.len() != before.len() || now.modified().ok() != before.modified().ok() {
        let _ = std::fs::remove_file(&tmp);
        bail!("{path:?} changed while it was being resealed; stop the server and run again");
    }
    std::fs::rename(&tmp, path)?;
    if let Some(dir) = path.parent() {
        OpenOptions::new().read(true).open(dir)?.sync_all()?;
    }
    report.rewritten = true;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent_log::IntentLog;

    fn key(byte: u8) -> SecretCryptor {
        SecretCryptor::new(zeroize::Zeroizing::new([byte; 32]))
    }

    fn tmp_log(stem: &str) -> PathBuf {
        let dir = tempfile::tempdir().unwrap().keep();
        dir.join(format!("{stem}.log"))
    }

    #[test]
    fn plaintext_frames_are_sealed_and_every_lsn_and_payload_survives() {
        let path = tmp_log("mixed");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"plaintext memory one").unwrap();
            log.append(b"plaintext memory two").unwrap();
            log.sync().unwrap();
        }
        {
            let mut log = IntentLog::open_with_cryptor(&path, Some(key(9))).unwrap();
            log.append(b"already sealed three").unwrap();
            log.sync().unwrap();
        }

        let dry = reseal_plaintext_frames(&path, &key(9), true).unwrap();
        assert_eq!(
            (dry.frames, dry.resealed, dry.already_sealed, dry.rewritten),
            (3, 2, 1, false)
        );
        let raw = std::fs::read(&path).unwrap();
        assert!(
            raw.windows(9).any(|w| w == b"plaintext"),
            "a dry run writes nothing"
        );

        let run = reseal_plaintext_frames(&path, &key(9), false).unwrap();
        assert_eq!((run.resealed, run.rewritten), (2, true));
        let raw = std::fs::read(&path).unwrap();
        assert!(
            !raw.windows(9).any(|w| w == b"plaintext"),
            "no plaintext frame is left"
        );

        let log = IntentLog::open_with_cryptor(&path, Some(key(9))).unwrap();
        let records: Vec<_> = log.iter().unwrap().map(|r| r.unwrap()).collect();
        let lsns: Vec<u64> = records.iter().map(|r| r.lsn.0).collect();
        assert_eq!(lsns, vec![0, 1, 2]);
        assert_eq!(records[1].payload, b"plaintext memory two");
        assert_eq!(records[2].payload, b"already sealed three");

        let again = reseal_plaintext_frames(&path, &key(9), false).unwrap();
        assert_eq!(
            (again.resealed, again.rewritten),
            (0, false),
            "a second run is a no-op"
        );
    }

    #[test]
    fn a_frame_sealed_under_another_key_refuses_and_leaves_the_log_alone() {
        let path = tmp_log("foreign");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"plaintext memory").unwrap();
            log.sync().unwrap();
        }
        {
            let mut log = IntentLog::open_with_cryptor(&path, Some(key(1))).unwrap();
            log.append(b"sealed elsewhere").unwrap();
            log.sync().unwrap();
        }
        let before = std::fs::read(&path).unwrap();
        assert!(reseal_plaintext_frames(&path, &key(2), false).is_err());
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "the log is unchanged"
        );
        assert!(
            !path.with_extension("intentlog.reseal.tmp").exists(),
            "no temporary left behind"
        );
    }
}
