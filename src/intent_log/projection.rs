//! Projection trait + replay driver.
//!
//! A projection is anything that derives its state from the intent log:
//! the Vamana vector index, the BM25 inverted index, the SQLite gap-topology
//! store, eventually Postgres (W4). Each projection records how far it has
//! applied via a [`Lsn`] checkpoint; the replay driver resumes from
//! `checkpoint().next()` to the head of the log.
//!
//! ## Idempotency contract
//!
//! `apply()` MUST be idempotent. Replaying the same record twice (after a
//! checkpoint failed to persist, say) must produce the same state as
//! applying it once. The simplest way to satisfy this is to key writes by
//! the record's LSN: an UPSERT keyed by `(projection_name, lsn)` is
//! trivially idempotent.
//!
//! ## Crash recovery
//!
//! The driver assumes:
//!   - intent_log entries are durable before any projection sees them
//!     (writers `sync()` after `append()` for the records they care about);
//!   - a projection that fails mid-apply leaves the checkpoint at the LSN
//!     *before* the failing record, so on restart the driver re-applies it
//!     (and idempotency makes that safe).

use std::error::Error;

use super::{IntentLog, IntentLogError, IntentRecord, Lsn};

/// Trait implemented by every store that derives its state from the intent
/// log. See module docs for the idempotency and crash-recovery contract.
pub trait Projection {
    type Error: Error + Send + Sync + 'static;

    /// Stable, unique identifier for this projection. Used as the key when
    /// checkpoints are persisted to a shared store. Must be stable across
    /// process restarts and code versions — renaming this is a migration.
    fn name(&self) -> &str;

    /// Apply a single record. MUST be idempotent — see module docs.
    fn apply(&mut self, record: &IntentRecord) -> Result<(), Self::Error>;

    /// LSN of the last record successfully applied. The driver resumes from
    /// `checkpoint().map(Lsn::next).unwrap_or(Lsn::ZERO)`. Returning
    /// `None` means the projection has never applied anything and should
    /// replay from the start of the log.
    fn checkpoint(&self) -> Option<Lsn>;

    /// Persist the in-memory checkpoint so that a crash followed by
    /// `checkpoint()` returns the same value. The driver calls this after
    /// each batch (or at the end of replay); cheap checkpoint stores can
    /// no-op until shutdown, expensive ones can persist incrementally.
    fn persist_checkpoint(&mut self) -> Result<(), Self::Error>;
}

/// Errors raised by the replay driver. Distinguishes log-side problems
/// (corrupt frames, I/O) from projection-side problems (the projection's
/// own `apply` returned an error).
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    #[error("intent log error: {0}")]
    Log(#[from] IntentLogError),

    #[error("projection '{projection}' failed at lsn {lsn:?}: {source}")]
    ProjectionFailed {
        projection: String,
        lsn: Lsn,
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },
}

/// Replay every record in `log` whose LSN is greater than the projection's
/// current checkpoint. Returns the number of records applied.
///
/// If `persist_every` is `Some(n)`, the driver calls `persist_checkpoint()`
/// after every `n` successfully-applied records. Either way, on clean exit
/// the checkpoint is persisted one final time.
///
/// If `apply()` returns an error, replay stops and the driver returns
/// without advancing past the failing record. The projection's checkpoint
/// will reflect the last *successful* apply, so on restart the failed
/// record is re-attempted (idempotency makes that safe).
pub fn replay<P: Projection>(
    log: &IntentLog,
    projection: &mut P,
    persist_every: Option<u64>,
) -> Result<u64, ReplayError> {
    let resume_at = projection
        .checkpoint()
        .map(Lsn::next)
        .unwrap_or(Lsn::ZERO);

    let mut applied = 0u64;
    let mut last_persisted_at = 0u64;

    for frame in log.iter()? {
        let record = frame?;
        if record.lsn < resume_at {
            continue;
        }
        projection
            .apply(&record)
            .map_err(|e| ReplayError::ProjectionFailed {
                projection: projection.name().to_string(),
                lsn: record.lsn,
                source: Box::new(e),
            })?;
        applied += 1;

        if let Some(n) = persist_every {
            if applied - last_persisted_at >= n {
                projection
                    .persist_checkpoint()
                    .map_err(|e| ReplayError::ProjectionFailed {
                        projection: projection.name().to_string(),
                        lsn: record.lsn,
                        source: Box::new(e),
                    })?;
                last_persisted_at = applied;
            }
        }
    }

    projection
        .persist_checkpoint()
        .map_err(|e| ReplayError::ProjectionFailed {
            projection: projection.name().to_string(),
            lsn: projection.checkpoint().unwrap_or(Lsn::ZERO),
            source: Box::new(e),
        })?;

    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::path::PathBuf;

    /// In-memory projection that appends every applied payload to a Vec and
    /// updates its in-memory checkpoint. Used to exercise the driver
    /// without touching disk.
    struct VecProjection {
        name: String,
        applied: Vec<(Lsn, Vec<u8>)>,
        checkpoint: Option<Lsn>,
        persisted_checkpoint: Option<Lsn>,
    }

    impl VecProjection {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                applied: Vec::new(),
                checkpoint: None,
                persisted_checkpoint: None,
            }
        }
    }

    impl Projection for VecProjection {
        type Error = Infallible;

        fn name(&self) -> &str {
            &self.name
        }

        fn apply(&mut self, record: &IntentRecord) -> Result<(), Infallible> {
            self.applied.push((record.lsn, record.payload.clone()));
            self.checkpoint = Some(record.lsn);
            Ok(())
        }

        fn checkpoint(&self) -> Option<Lsn> {
            self.checkpoint
        }

        fn persist_checkpoint(&mut self) -> Result<(), Infallible> {
            self.persisted_checkpoint = self.checkpoint;
            Ok(())
        }
    }

    /// Projection whose `apply` fails the first time it sees a given LSN
    /// but succeeds on a retry — exercises the "idempotent re-apply"
    /// crash-recovery path.
    struct FlakyProjection {
        name: String,
        applied: Vec<Lsn>,
        checkpoint: Option<Lsn>,
        fail_once_at: Option<Lsn>,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("flaky projection failed: {0}")]
    struct FlakyError(String);

    impl Projection for FlakyProjection {
        type Error = FlakyError;

        fn name(&self) -> &str {
            &self.name
        }

        fn apply(&mut self, record: &IntentRecord) -> Result<(), FlakyError> {
            if Some(record.lsn) == self.fail_once_at {
                self.fail_once_at = None;
                return Err(FlakyError(format!("seeded failure at lsn {:?}", record.lsn)));
            }
            self.applied.push(record.lsn);
            self.checkpoint = Some(record.lsn);
            Ok(())
        }

        fn checkpoint(&self) -> Option<Lsn> {
            self.checkpoint
        }

        fn persist_checkpoint(&mut self) -> Result<(), FlakyError> {
            Ok(())
        }
    }

    fn tmp_log_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("veld-intent-log-projection-{name}-{pid}-{stamp}.log"));
        p
    }

    #[test]
    fn replay_from_scratch_applies_every_record() {
        let path = tmp_log_path("scratch");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"one").unwrap();
            log.append(b"two").unwrap();
            log.append(b"three").unwrap();
            log.sync().unwrap();
        }

        let log = IntentLog::open(&path).unwrap();
        let mut proj = VecProjection::new("test");
        let applied = replay(&log, &mut proj, None).unwrap();

        assert_eq!(applied, 3);
        assert_eq!(proj.applied.len(), 3);
        assert_eq!(proj.applied[0], (Lsn(0), b"one".to_vec()));
        assert_eq!(proj.applied[2], (Lsn(2), b"three".to_vec()));
        assert_eq!(proj.persisted_checkpoint, Some(Lsn(2)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn replay_resumes_from_checkpoint() {
        let path = tmp_log_path("resume");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"already-applied-0").unwrap();
            log.append(b"already-applied-1").unwrap();
            log.append(b"new-2").unwrap();
            log.append(b"new-3").unwrap();
            log.sync().unwrap();
        }

        let log = IntentLog::open(&path).unwrap();
        let mut proj = VecProjection::new("test");
        proj.checkpoint = Some(Lsn(1));
        proj.persisted_checkpoint = Some(Lsn(1));

        let applied = replay(&log, &mut proj, None).unwrap();
        assert_eq!(applied, 2);
        assert_eq!(proj.applied.len(), 2);
        assert_eq!(proj.applied[0].0, Lsn(2));
        assert_eq!(proj.applied[1].0, Lsn(3));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn persist_every_n_calls_persist_at_threshold() {
        let path = tmp_log_path("persist_every");
        {
            let mut log = IntentLog::open(&path).unwrap();
            for i in 0..7u8 {
                log.append(&[i]).unwrap();
            }
            log.sync().unwrap();
        }

        // Wrapper that counts persist calls.
        struct CountingProjection {
            inner: VecProjection,
            persist_calls: u32,
        }
        impl Projection for CountingProjection {
            type Error = Infallible;
            fn name(&self) -> &str {
                self.inner.name()
            }
            fn apply(&mut self, r: &IntentRecord) -> Result<(), Infallible> {
                self.inner.apply(r)
            }
            fn checkpoint(&self) -> Option<Lsn> {
                self.inner.checkpoint()
            }
            fn persist_checkpoint(&mut self) -> Result<(), Infallible> {
                self.persist_calls += 1;
                self.inner.persist_checkpoint()
            }
        }

        let log = IntentLog::open(&path).unwrap();
        let mut proj = CountingProjection {
            inner: VecProjection::new("counter"),
            persist_calls: 0,
        };
        replay(&log, &mut proj, Some(3)).unwrap();

        // 7 records, persist_every=3 → persist after #3 and #6, plus
        // one final on clean exit. That's 3 calls.
        assert_eq!(proj.persist_calls, 3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn replay_stops_on_projection_error_and_keeps_checkpoint() {
        let path = tmp_log_path("flaky");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"a").unwrap(); // lsn 0
            log.append(b"b").unwrap(); // lsn 1
            log.append(b"c").unwrap(); // lsn 2  ← fail here
            log.append(b"d").unwrap(); // lsn 3
            log.sync().unwrap();
        }

        let log = IntentLog::open(&path).unwrap();
        let mut proj = FlakyProjection {
            name: "flaky".to_string(),
            applied: Vec::new(),
            checkpoint: None,
            fail_once_at: Some(Lsn(2)),
        };
        let err = replay(&log, &mut proj, None).unwrap_err();
        match err {
            ReplayError::ProjectionFailed { lsn, .. } => assert_eq!(lsn, Lsn(2)),
            other => panic!("unexpected error: {other:?}"),
        }

        // Checkpoint is at lsn 1 — the last *successful* apply.
        assert_eq!(proj.checkpoint, Some(Lsn(1)));
        assert_eq!(proj.applied, vec![Lsn(0), Lsn(1)]);

        // Restart: same projection, resumes from Lsn(2). The seeded failure
        // is now exhausted, so apply succeeds. Idempotent re-apply path.
        let applied = replay(&log, &mut proj, None).unwrap();
        assert_eq!(applied, 2);
        assert_eq!(proj.applied, vec![Lsn(0), Lsn(1), Lsn(2), Lsn(3)]);
        assert_eq!(proj.checkpoint, Some(Lsn(3)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn replay_returns_zero_when_log_is_empty() {
        let path = tmp_log_path("empty_replay");
        let log = IntentLog::open(&path).unwrap();
        let mut proj = VecProjection::new("empty");
        let applied = replay(&log, &mut proj, None).unwrap();
        assert_eq!(applied, 0);
        assert!(proj.applied.is_empty());
        // Persist still ran once on clean exit so the (None) checkpoint is
        // explicitly recorded.
        assert_eq!(proj.persisted_checkpoint, None);
        let _ = std::fs::remove_file(&path);
    }
}
