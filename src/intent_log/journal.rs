//! Writer-side primitive that ties log-append + sync + projection-apply
//! into one operation.
//!
//! ## The "log first" invariant
//!
//! Every state-changing operation flows through one path:
//!
//! ```text
//!   1. encode payload → bytes
//!   2. log.append(bytes)              ← assigns LSN
//!   3. log.sync()                     ← durable
//!   4. for each projection: apply
//!   5. record any apply errors and return
//! ```
//!
//! After step 3 the operation is *committed* — survives a crash. If any
//! projection's `apply` (step 4) fails, the failure is collected but the
//! whole `record_and_apply` call still reports success, because the log
//! holds the record and the replay driver will re-apply on restart. This
//! is what the `Projection::apply` idempotency contract is for: replay
//! after a crashed apply must not double-count.
//!
//! ## Why this lives here, not inline at the call site
//!
//! Doing the log+sync+apply dance by hand at every memory CRUD call site
//! is exactly how `health_index` ended up as the unfixed twin of
//! `health_ready` — copy-paste drift across paths that should share a
//! structural rule. One writer primitive means one place to audit, one
//! place to fix, one place to add metrics.

use std::error::Error;

use crate::metrics::{PROJECTION_APPLY_DURATION, PROJECTION_APPLY_TOTAL, Timer};

use super::payload::{self, IntentPayload, PayloadError};
use super::{IntentLog, IntentLogError, Lsn};

/// Object-safe variant of `Projection` that takes a *typed* payload + the
/// LSN it was assigned, instead of a raw `IntentRecord`. The static
/// `Projection` trait still exists for replay-time consumers (which work
/// from `IntentRecord` straight off the log); this dyn-friendly variant
/// is what `JournaledWriter` calls during normal writes.
///
/// Implementors typically own a single in-memory checkpoint and a single
/// `CheckpointStore` reference. They should be cheap to keep in a list
/// inside `JournaledWriter`.
pub trait TypedProjection: Send {
    /// Stable, unique identifier for this projection.
    fn name(&self) -> &str;

    /// Apply a typed payload at a given LSN. MUST be idempotent — see
    /// `super::projection` module docs.
    fn apply(
        &mut self,
        lsn: Lsn,
        payload: &IntentPayload,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
}

/// Errors raised by [`JournaledWriter::record_and_apply`].
#[derive(Debug, thiserror::Error)]
pub enum JournalError {
    #[error("intent log error: {0}")]
    Log(#[from] IntentLogError),
    #[error("payload encode error: {0}")]
    Payload(#[from] PayloadError),
}

/// Outcome of a successful `record_and_apply`. The LSN is the log
/// position that was assigned; `apply_errors` holds any per-projection
/// failures (these do not fail the call — the log entry survives, and
/// replay will retry on restart).
#[derive(Debug)]
pub struct WriteOutcome {
    pub lsn: Lsn,
    pub apply_errors: Vec<ApplyError>,
}

/// One projection's failed apply during a journaled write. Carried in
/// `WriteOutcome.apply_errors` so the caller can log/metrics-emit per
/// projection without taking down the whole write path.
#[derive(Debug)]
pub struct ApplyError {
    pub projection: String,
    pub source: Box<dyn Error + Send + Sync>,
}

/// Ties an [`IntentLog`] together with a list of [`TypedProjection`]s and
/// orchestrates the log-first write pattern.
pub struct JournaledWriter {
    log: IntentLog,
    projections: Vec<Box<dyn TypedProjection>>,
}

impl JournaledWriter {
    /// Create a writer around an open log. Projections are added with
    /// [`add_projection`]; ordering only matters in that earlier
    /// projections see `apply` calls before later ones within the same
    /// `record_and_apply` invocation.
    pub fn new(log: IntentLog) -> Self {
        Self {
            log,
            projections: Vec::new(),
        }
    }

    pub fn add_projection(&mut self, p: Box<dyn TypedProjection>) {
        self.projections.push(p);
    }

    /// Number of projections currently attached.
    pub fn projection_count(&self) -> usize {
        self.projections.len()
    }

    /// Borrow the underlying log read-only. Useful for diagnostics —
    /// don't append directly to it, that bypasses the projection
    /// dispatch and breaks the log-first invariant.
    pub fn log(&self) -> &IntentLog {
        &self.log
    }

    /// The whole-system contract:
    ///
    ///   1. Encode `payload` to bytes.
    ///   2. Append to the log.
    ///   3. `sync` (fsync the underlying file).
    ///   4. Walk every projection and call `apply(lsn, payload)`.
    ///
    /// If steps 1-3 succeed, returns `Ok(WriteOutcome)`. Any errors from
    /// step 4 are collected into `WriteOutcome.apply_errors` — they do
    /// NOT fail the call. The log holds the record, so the replay
    /// driver will retry on restart; idempotency in `apply` makes that
    /// safe.
    ///
    /// If step 1, 2, or 3 fails, the call returns `Err` and no
    /// projection is touched. The log may have a partial frame in this
    /// case; `IntentLog::open` on the next process will detect it via
    /// CRC mismatch / truncated frame and `truncate_corrupt_tail()`
    /// will clear it.
    #[tracing::instrument(
        level = "info",
        name = "journal.record_and_apply",
        skip(self, payload),
        fields(
            user_id = %payload.user_id(),
            memory_id = %payload.memory_id(),
            projection_count = self.projections.len(),
            lsn = tracing::field::Empty,
        ),
    )]
    pub fn record_and_apply(
        &mut self,
        payload: &IntentPayload,
    ) -> Result<WriteOutcome, JournalError> {
        let lsn = match payload::append(&mut self.log, payload) {
            Ok(lsn) => lsn,
            Err(PayloadError::Log(e)) => {
                tracing::error!(
                    error = %e,
                    "journal record_and_apply failed: intent log error during append",
                );
                return Err(JournalError::Log(e));
            }
            Err(
                e @ (PayloadError::Encode(_)
                | PayloadError::Decode(_)
                | PayloadError::UnknownSchemaVersion { .. }
                | PayloadError::Migration(_)),
            ) => {
                // Encode-side: a payload variant this binary cannot
                // serialise (rare — bincode of #[non_exhaustive] enums
                // does not fail in practice). Decode-side and the two
                // schema-version variants are not reachable from
                // `payload::append` today (that function only encodes)
                // but we cover them so adding a new payload error
                // variant upstream is a "fix the match" change, not a
                // compile-error scramble at every call site.
                tracing::error!(
                    error = %e,
                    "journal record_and_apply failed: payload encode error",
                );
                return Err(JournalError::Payload(e));
            }
        };
        tracing::Span::current().record("lsn", lsn.0);
        if let Err(e) = self.log.sync() {
            tracing::error!(
                error = %e,
                lsn = lsn.0,
                "journal record_and_apply failed: intent log sync error",
            );
            return Err(JournalError::Log(e));
        }

        let mut apply_errors = Vec::new();
        for p in &mut self.projections {
            let projection_name = p.name().to_string();
            let timer = Timer::new(
                PROJECTION_APPLY_DURATION
                    .with_label_values(&[projection_name.as_str()]),
            );
            let apply_result = p.apply(lsn, payload);
            drop(timer);
            match apply_result {
                Ok(()) => {
                    PROJECTION_APPLY_TOTAL
                        .with_label_values(&[projection_name.as_str(), "ok"])
                        .inc();
                }
                Err(source) => {
                    PROJECTION_APPLY_TOTAL
                        .with_label_values(&[projection_name.as_str(), "error"])
                        .inc();
                    tracing::error!(
                        projection = %projection_name,
                        lsn = lsn.0,
                        error = %source,
                        "projection apply failed during journaled write",
                    );
                    apply_errors.push(ApplyError {
                        projection: projection_name,
                        source,
                    });
                }
            }
        }

        Ok(WriteOutcome { lsn, apply_errors })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    /// In-memory projection that records every (lsn, payload) it sees.
    struct RecordingProjection {
        name: String,
        applied: Arc<Mutex<Vec<(Lsn, IntentPayload)>>>,
    }

    impl TypedProjection for RecordingProjection {
        fn name(&self) -> &str {
            &self.name
        }
        fn apply(
            &mut self,
            lsn: Lsn,
            payload: &IntentPayload,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            self.applied.lock().unwrap().push((lsn, payload.clone()));
            Ok(())
        }
    }

    /// Projection that always fails on apply, so we can verify the
    /// "errors collected, write still succeeds" pathway.
    struct FailingProjection {
        name: String,
    }
    impl TypedProjection for FailingProjection {
        fn name(&self) -> &str {
            &self.name
        }
        fn apply(
            &mut self,
            _lsn: Lsn,
            _payload: &IntentPayload,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            Err("synthetic failure".into())
        }
    }

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("veld-journal-{name}-{pid}-{stamp}.log"));
        p
    }

    #[test]
    fn record_and_apply_dispatches_to_every_projection() {
        let path = tmp_path("dispatch");
        let log = IntentLog::open(&path).unwrap();
        let mut writer = JournaledWriter::new(log);

        let a_seen = Arc::new(Mutex::new(Vec::new()));
        let b_seen = Arc::new(Mutex::new(Vec::new()));
        writer.add_projection(Box::new(RecordingProjection {
            name: "a".into(),
            applied: a_seen.clone(),
        }));
        writer.add_projection(Box::new(RecordingProjection {
            name: "b".into(),
            applied: b_seen.clone(),
        }));

        let payload = IntentPayload::Forget {
            user_id: "u".into(),
            memory_id: "m-1".into(),
            schema_version: None,
        };
        let outcome = writer.record_and_apply(&payload).unwrap();
        assert_eq!(outcome.lsn, Lsn(0));
        assert!(outcome.apply_errors.is_empty());

        let a = a_seen.lock().unwrap();
        let b = b_seen.lock().unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_eq!(a[0], (Lsn(0), payload.clone()));
        assert_eq!(b[0], (Lsn(0), payload));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn lsns_advance_monotonically_across_calls() {
        let path = tmp_path("lsn_mono");
        let log = IntentLog::open(&path).unwrap();
        let mut writer = JournaledWriter::new(log);
        let seen = Arc::new(Mutex::new(Vec::new()));
        writer.add_projection(Box::new(RecordingProjection {
            name: "rec".into(),
            applied: seen.clone(),
        }));

        let payloads: Vec<IntentPayload> = (0..5)
            .map(|i| IntentPayload::Anchor {
                user_id: "u".into(),
                memory_id: format!("m-{i}"),
                importance: 0.5,
                schema_version: None,
            })
            .collect();
        for p in &payloads {
            writer.record_and_apply(p).unwrap();
        }
        let seen = seen.lock().unwrap();
        let lsns: Vec<_> = seen.iter().map(|(l, _)| *l).collect();
        assert_eq!(lsns, vec![Lsn(0), Lsn(1), Lsn(2), Lsn(3), Lsn(4)]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn failing_projection_does_not_fail_the_write() {
        let path = tmp_path("apply_err");
        let log = IntentLog::open(&path).unwrap();
        let mut writer = JournaledWriter::new(log);
        let good_seen = Arc::new(Mutex::new(Vec::new()));
        writer.add_projection(Box::new(FailingProjection {
            name: "bad".into(),
        }));
        writer.add_projection(Box::new(RecordingProjection {
            name: "good".into(),
            applied: good_seen.clone(),
        }));

        let payload = IntentPayload::Remember {
            user_id: "u".into(),
            memory_id: "m".into(),
            memory_bincode: vec![1, 2, 3],
            schema_version: None,
        };
        let outcome = writer.record_and_apply(&payload).unwrap();
        assert_eq!(outcome.lsn, Lsn(0));
        // Bad projection collected its error.
        assert_eq!(outcome.apply_errors.len(), 1);
        assert_eq!(outcome.apply_errors[0].projection, "bad");
        // Good projection still ran.
        assert_eq!(good_seen.lock().unwrap().len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn log_holds_the_record_even_when_every_projection_fails() {
        let path = tmp_path("log_survives");
        {
            let log = IntentLog::open(&path).unwrap();
            let mut writer = JournaledWriter::new(log);
            writer.add_projection(Box::new(FailingProjection {
                name: "x".into(),
            }));
            let payload = IntentPayload::Forget {
                user_id: "u".into(),
                memory_id: "m".into(),
                schema_version: None,
            };
            let outcome = writer.record_and_apply(&payload).unwrap();
            assert_eq!(outcome.apply_errors.len(), 1);
        }
        // Reopen and confirm the record is durable.
        let log = IntentLog::open(&path).unwrap();
        let records: Vec<_> = log.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn projection_count_reports_attached_count() {
        let path = tmp_path("count");
        let log = IntentLog::open(&path).unwrap();
        let mut w = JournaledWriter::new(log);
        assert_eq!(w.projection_count(), 0);
        w.add_projection(Box::new(FailingProjection { name: "a".into() }));
        w.add_projection(Box::new(FailingProjection { name: "b".into() }));
        assert_eq!(w.projection_count(), 2);
        let _ = std::fs::remove_file(&path);
    }
}
