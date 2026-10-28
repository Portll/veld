//! Erasure of forgotten memories from sealed journal segments.
//!
//! A hard forget journals a `Forget` frame and leaves every earlier frame of
//! that memory in the log: its content stays on disk for as long as the log
//! does. Once a `Forget` is old enough (it sits in a sealed segment last
//! written at least `min_age` ago), [`purge_forgotten`] rewrites the sealed
//! segments without that memory's frames up to and including the `Forget`.
//! Replay from the start still reaches the same live state, since the memory
//! never appears, and its content leaves the live journal. A memory created
//! again under the same id after the `Forget` keeps its later frames.
//!
//! The head file is never touched, so appends go on undisturbed. A rewritten
//! segment goes to a fresh file renamed over the old one: a backup's hard
//! link keeps the frames it captured until that backup is purged, and an open
//! reader finishes the copy it started on. The rewrite carries
//! [`super::header::FLAG_SPARSE`], so readers accept the LSNs it skips, and
//! keeps the old file's modification time, which is the segment's age.
//!
//! The window must outlast any replication follower's time behind the
//! leader: a follower that has applied a memory but not yet read its `Forget`
//! never sees that `Forget` once it is purged, and keeps the memory.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::header::FLAG_SPARSE;
use super::{
    sealed_segments, sync_parent_dir, IntentLogError, IntentLogHeader, IntentLogIter,
    IntentPayload, Lsn, Segment, HEADER_BYTES,
};
use crate::keystore::SecretCryptor;

/// What one [`purge_forgotten`] pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct PurgeReport {
    /// Sealed segments old enough to be considered.
    pub segments_eligible: usize,
    /// Segments rewritten without purged frames.
    pub segments_rewritten: usize,
    /// Memories whose frames were purged.
    pub memories_purged: usize,
    pub frames_dropped: u64,
    pub bytes_reclaimed: u64,
}

/// Rewrite the sealed segments of the log at `log_path` without the frames of
/// every memory whose `Forget` lies in a segment last written at least
/// `min_age` before `now`, and, when `follower_floor` is set, at or below it:
/// the lowest LSN a replication follower has read past, so no follower misses
/// a `Forget` it has not read. `cryptor` is the journal's key, needed to read
/// the memory ids of a sealed journal; frames it cannot read are kept.
pub fn purge_forgotten(
    log_path: &Path,
    cryptor: Option<&SecretCryptor>,
    min_age: Duration,
    follower_floor: Option<Lsn>,
    now: SystemTime,
) -> Result<PurgeReport, IntentLogError> {
    let cutoff = now.checked_sub(min_age).unwrap_or(SystemTime::UNIX_EPOCH);
    let mut eligible = Vec::new();
    for segment in sealed_segments(log_path)? {
        // Segments are sealed in LSN order, so the old ones are a prefix.
        if std::fs::metadata(&segment.path)?.modified()? > cutoff {
            break;
        }
        eligible.push(segment);
    }
    let mut report = PurgeReport {
        segments_eligible: eligible.len(),
        ..PurgeReport::default()
    };

    let mut forgotten: HashMap<String, Lsn> = HashMap::new();
    for segment in &eligible {
        scan(segment, cryptor, |lsn, payload| {
            if let Some(IntentPayload::Forget { memory_id, .. }) = payload {
                if follower_floor.is_none_or(|floor| lsn <= floor) {
                    forgotten.insert(memory_id.clone(), lsn);
                }
            }
            Ok(())
        })?;
    }
    if forgotten.is_empty() {
        return Ok(report);
    }

    let mut purged_ids = std::collections::HashSet::new();
    for segment in &eligible {
        if let Some((dropped, reclaimed)) = rewrite(segment, cryptor, &forgotten, &mut purged_ids)?
        {
            report.segments_rewritten += 1;
            report.frames_dropped += dropped;
            report.bytes_reclaimed += reclaimed;
        }
    }
    report.memories_purged = purged_ids.len();
    Ok(report)
}

/// Visit every frame of `segment` with its decoded payload (`None` when it
/// does not decode, or cannot be opened without a key).
fn scan(
    segment: &Segment,
    cryptor: Option<&SecretCryptor>,
    mut visit: impl FnMut(Lsn, Option<&IntentPayload>) -> Result<(), IntentLogError>,
) -> Result<(), IntentLogError> {
    let mut frames = IntentLogIter::open(&segment.path, segment.frame_start, None)?;
    while let Some(meta) = frames.read_frame()? {
        let payload = open_payload(frames.sealed_payload(), cryptor);
        visit(meta.lsn, payload.as_ref())?;
    }
    Ok(())
}

fn open_payload(sealed: &[u8], cryptor: Option<&SecretCryptor>) -> Option<IntentPayload> {
    let plaintext = match cryptor {
        Some(c) if SecretCryptor::is_encrypted(sealed) => c.decrypt_bytes(sealed, b"").ok()?,
        _ if SecretCryptor::is_encrypted(sealed) => return None,
        _ => sealed.to_vec(),
    };
    super::payload::decode(&plaintext).ok()
}

/// Rewrite `segment` without the frames of memories in `forgotten` up to their
/// `Forget`. `None` when it holds none of them; otherwise the frames dropped
/// and the bytes reclaimed.
fn rewrite(
    segment: &Segment,
    cryptor: Option<&SecretCryptor>,
    forgotten: &HashMap<String, Lsn>,
    purged_ids: &mut std::collections::HashSet<String>,
) -> Result<Option<(u64, u64)>, IntentLogError> {
    let drops = |lsn: Lsn, payload: Option<&IntentPayload>| {
        payload
            .and_then(|p| forgotten.get(p.memory_id()).map(|forget| (p, forget)))
            .filter(|(_, forget)| lsn <= **forget)
            .map(|(p, _)| p.memory_id().to_string())
    };
    if segment.frame_start == 0 {
        // A legacy file has no header to carry the sparse flag, and giving it
        // one would move its frames under the offsets the live log holds.
        tracing::warn!(
            segment = %segment.path.display(),
            "legacy intent log segment holds no header; its forgotten frames are kept",
        );
        return Ok(None);
    }
    let mut any = false;
    scan(segment, cryptor, |lsn, payload| {
        any |= drops(lsn, payload).is_some();
        Ok(())
    })?;
    if !any {
        return Ok(None);
    }

    let before = std::fs::metadata(&segment.path)?;
    let staged = staging_path(&segment.path);
    let mut header = read_header(&segment.path)?;
    header.flags |= FLAG_SPARSE;
    let mut dropped = 0u64;
    {
        let mut out = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&staged)?,
        );
        out.write_all(&header.to_bytes())?;
        let mut frames = IntentLogIter::open(&segment.path, segment.frame_start, None)?;
        while let Some(meta) = frames.read_frame()? {
            let sealed = frames.sealed_payload();
            if let Some(id) = drops(meta.lsn, open_payload(sealed, cryptor).as_ref()) {
                purged_ids.insert(id);
                dropped += 1;
                continue;
            }
            // Copied as stored: the CRC covers length, LSN and sealed bytes,
            // none of which change.
            out.write_all(&(meta.len as u32).to_le_bytes())?;
            out.write_all(&meta.lsn.0.to_le_bytes())?;
            out.write_all(sealed)?;
            out.write_all(&meta.crc.to_le_bytes())?;
        }
        out.flush()?;
        let file = out.into_inner().map_err(|e| e.into_error())?;
        file.sync_data()?;
        file.set_modified(before.modified()?)?;
    }
    let after = std::fs::metadata(&staged)?.len();
    std::fs::rename(&staged, &segment.path)?;
    sync_parent_dir(&segment.path)?;
    // A backup's cached record of the old file no longer describes it.
    let mut record = segment.path.as_os_str().to_os_string();
    record.push(".backup.json");
    let _ = std::fs::remove_file(PathBuf::from(record));
    tracing::info!(
        segment = %segment.path.display(),
        dropped,
        bytes_before = before.len(),
        bytes_after = after,
        "intent log segment rewritten without the frames of forgotten memories",
    );
    Ok(Some((dropped, before.len().saturating_sub(after))))
}

fn read_header(path: &Path) -> Result<IntentLogHeader, IntentLogError> {
    let mut file = File::open(path)?;
    let mut buf = [0u8; HEADER_BYTES];
    std::io::Read::read_exact(&mut file, &mut buf)?;
    Ok(IntentLogHeader::parse(&buf)?)
}

fn staging_path(segment: &Path) -> PathBuf {
    let mut name = segment.as_os_str().to_os_string();
    name.push(".purge.tmp");
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent_log::{payload, IntentLog};

    const DAY: Duration = Duration::from_secs(86_400);

    fn remember(id: &str, body: &str) -> Vec<u8> {
        payload::encode(&IntentPayload::Remember {
            user_id: "tenant".to_string(),
            memory_id: id.to_string(),
            memory_bincode: body.as_bytes().to_vec(),
            schema_version: None,
        })
        .unwrap()
    }

    fn forget(id: &str) -> Vec<u8> {
        payload::encode(&IntentPayload::Forget {
            user_id: "tenant".to_string(),
            memory_id: id.to_string(),
            schema_version: None,
        })
        .unwrap()
    }

    /// `a` remembered, updated and forgotten, with `b` and `c` around it, each
    /// frame synced so that a 120-byte head rotates every couple of frames.
    fn written(dir: &tempfile::TempDir, cryptor: Option<SecretCryptor>) -> (IntentLog, Lsn) {
        let mut log = IntentLog::open_with_cryptor(dir.path().join("intent.log"), cryptor).unwrap();
        log.set_segment_bytes(Some(120));
        let mut forget_lsn = Lsn::ZERO;
        for frame in [
            remember("a", "alpha-secret"),
            remember("b", "bravo"),
            remember("a", "alpha-secret-revised"),
            forget("a"),
            remember("c", "charlie"),
            remember("b", "bravo-revised"),
        ] {
            let lsn = log.append(&frame).unwrap();
            if frame == forget("a") {
                forget_lsn = lsn;
            }
            log.sync().unwrap();
        }
        assert!(
            log.base_lsn() > forget_lsn,
            "the Forget is sealed, not in the head"
        );
        (log, forget_lsn)
    }

    fn age_sealed(log: &IntentLog, by: Duration) {
        for segment in log.sealed() {
            OpenOptions::new()
                .write(true)
                .open(&segment.path)
                .unwrap()
                .set_modified(SystemTime::now() - by)
                .unwrap();
        }
    }

    fn memory_ids(log: &IntentLog) -> Vec<(u64, String)> {
        log.iter()
            .unwrap()
            .map(|r| {
                let r = r.unwrap();
                (
                    r.lsn.0,
                    payload::decode(&r.payload).unwrap().memory_id().to_string(),
                )
            })
            .collect()
    }

    fn journal_bytes(dir: &tempfile::TempDir) -> Vec<u8> {
        crate::intent_log::log_files(&dir.path().join("intent.log"))
            .unwrap()
            .iter()
            .flat_map(|f| std::fs::read(f).unwrap())
            .collect()
    }

    fn contains(haystack: &[u8], needle: &str) -> bool {
        haystack
            .windows(needle.len())
            .any(|w| w == needle.as_bytes())
    }

    #[test]
    fn a_forget_past_the_window_takes_its_memory_out_of_the_journal() {
        let dir = tempfile::TempDir::new().unwrap();
        let (mut log, forget_lsn) = written(&dir, None);
        age_sealed(&log, 10 * DAY);
        let first = log.sealed()[0].path.clone();
        let linked = dir.path().join("backup-link");
        std::fs::hard_link(&first, &linked).unwrap();
        let next = log.next_lsn();

        let report = purge_forgotten(log.path(), None, 7 * DAY, None, SystemTime::now()).unwrap();

        assert_eq!(report.memories_purged, 1);
        assert_eq!(report.frames_dropped, 3, "two snapshots and the Forget");
        assert!(!contains(&journal_bytes(&dir), "alpha-secret"));
        assert!(
            contains(&std::fs::read(&linked).unwrap(), "alpha-secret"),
            "a backup's hard link keeps what it captured"
        );
        let kept = memory_ids(&log);
        assert!(kept.iter().all(|(_, id)| id != "a"), "{kept:?}");
        assert_eq!(kept.iter().filter(|(_, id)| id == "b").count(), 2);
        assert!(kept.iter().all(|(lsn, _)| *lsn != forget_lsn.0));

        log.append(&remember("a", "alpha-again")).unwrap();
        log.sync().unwrap();
        assert_eq!(log.next_lsn(), next.next());
        let reopened = IntentLog::open(log.path()).unwrap();
        assert_eq!(reopened.next_lsn(), next.next());
        let after = memory_ids(&reopened);
        assert_eq!(
            after.last(),
            Some(&(next.0, "a".to_string())),
            "a later memory under the id stays"
        );

        let again = purge_forgotten(log.path(), None, 7 * DAY, None, SystemTime::now()).unwrap();
        assert_eq!(again.frames_dropped, 0);
        assert_eq!(again.segments_rewritten, 0);
    }

    /// A Forget past the lowest cursor a replication follower has read past is
    /// kept, so that follower still reads it; once followers are past it, it goes.
    #[test]
    fn a_forget_no_follower_has_read_yet_is_kept() {
        let dir = tempfile::TempDir::new().unwrap();
        let (log, forget_lsn) = written(&dir, None);
        age_sealed(&log, 10 * DAY);
        let behind = Lsn(forget_lsn.0 - 1);

        let held =
            purge_forgotten(log.path(), None, 7 * DAY, Some(behind), SystemTime::now()).unwrap();
        assert_eq!(held.frames_dropped, 0);
        assert!(contains(&journal_bytes(&dir), "alpha-secret"));

        let read = purge_forgotten(
            log.path(),
            None,
            7 * DAY,
            Some(forget_lsn),
            SystemTime::now(),
        )
        .unwrap();
        assert_eq!(read.frames_dropped, 3);
        assert!(!contains(&journal_bytes(&dir), "alpha-secret"));
    }

    #[test]
    fn a_forget_inside_the_window_is_kept() {
        let dir = tempfile::TempDir::new().unwrap();
        let (log, _) = written(&dir, None);
        age_sealed(&log, 2 * DAY);
        let report = purge_forgotten(log.path(), None, 7 * DAY, None, SystemTime::now()).unwrap();
        assert_eq!(report.segments_eligible, 0);
        assert!(contains(&journal_bytes(&dir), "alpha-secret"));
    }

    #[test]
    fn a_sealed_journal_is_purged_only_with_its_key() {
        let key = || SecretCryptor::new(zeroize::Zeroizing::new([9u8; 32]));
        let dir = tempfile::TempDir::new().unwrap();
        let (log, _) = written(&dir, Some(key()));
        age_sealed(&log, 10 * DAY);

        let keyless = purge_forgotten(log.path(), None, 7 * DAY, None, SystemTime::now()).unwrap();
        assert_eq!(keyless.frames_dropped, 0, "frames it cannot read are kept");

        let keyed =
            purge_forgotten(log.path(), Some(&key()), 7 * DAY, None, SystemTime::now()).unwrap();
        assert_eq!(keyed.frames_dropped, 3);
        assert!(memory_ids(&log).iter().all(|(_, id)| id != "a"));
    }

    #[test]
    fn a_cut_at_a_purged_lsn_falls_after_the_frame_below_it() {
        let dir = tempfile::TempDir::new().unwrap();
        let (mut log, forget_lsn) = written(&dir, None);
        age_sealed(&log, 10 * DAY);
        purge_forgotten(log.path(), None, 7 * DAY, None, SystemTime::now()).unwrap();

        log.truncate_to_lsn(forget_lsn).unwrap();
        let kept = memory_ids(&log);
        let last = kept.last().unwrap().0;
        assert!(last < forget_lsn.0, "{kept:?}");
        assert_eq!(log.next_lsn(), forget_lsn.next());
    }
}
