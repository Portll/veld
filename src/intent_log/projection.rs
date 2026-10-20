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

use std::collections::HashSet;
use std::error::Error;

use parking_lot::Mutex;

use crate::metrics::{
    Timer, CHECKPOINT_PERSIST_TOTAL, PROJECTION_APPLY_DURATION, PROJECTION_APPLY_TOTAL,
    PROJECTION_CHECKPOINT_LSN, PROJECTION_LAG_RECORDS, PROJECTION_REPLAY_RECORDS_TOTAL,
};

use super::checkpoint_store::{
    CheckpointStore, INTENT_LOG_SIDE_DATA, LIVE_MEMORY_IDS_KEY, MAX_SIDE_DATA_BYTES,
};
use super::payload::{IntentPayload, PayloadError};
use super::{FramePosition, IntentLog, IntentLogError, IntentRecord, Lsn, FRAME_POSITION_BYTES};

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

/// Object-safe view of a [`Projection`], so one replay pass can drive
/// projections of different concrete types. Implemented for every
/// `Projection`; the method names differ from `Projection`'s so both traits
/// can be in scope without ambiguity.
pub trait ReplayTarget {
    fn target_name(&self) -> &str;
    fn target_checkpoint(&self) -> Option<Lsn>;
    fn apply_record(&mut self, record: &IntentRecord) -> Result<(), BoxError>;
    fn persist_target_checkpoint(&mut self) -> Result<(), BoxError>;
}

type BoxError = Box<dyn Error + Send + Sync>;

impl<P: Projection> ReplayTarget for P {
    fn target_name(&self) -> &str {
        self.name()
    }

    fn target_checkpoint(&self) -> Option<Lsn> {
        self.checkpoint()
    }

    fn apply_record(&mut self, record: &IntentRecord) -> Result<(), BoxError> {
        self.apply(record).map_err(|e| Box::new(e) as BoxError)
    }

    fn persist_target_checkpoint(&mut self) -> Result<(), BoxError> {
        self.persist_checkpoint()
            .map_err(|e| Box::new(e) as BoxError)
    }
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
    let mut targets: [&mut dyn ReplayTarget; 1] = [projection];
    replay_all(log, &mut targets, persist_every)?
        .pop()
        .unwrap_or(Ok(0))
}

/// Replay several projections in one pass over the log. Each frame is read
/// and CRC-checked once, and opened (decrypted) only when at least one
/// projection still needs it; it is then applied to every projection whose
/// checkpoint is below its LSN, in slice order. Reading starts at the indexed
/// frame nearest the lowest checkpoint, not at the first frame.
///
/// Checkpoints stay independent: a projection whose `apply` or checkpoint
/// persist fails stops receiving records at that LSN (its checkpoint stays at
/// its last successful apply, so a restart retries the record) while the
/// others replay to the head. The same `persist_every` cadence and final
/// persist as [`replay`] apply to each projection.
///
/// Returns one result per projection, index-aligned with `targets`: the
/// number of records applied, or the failure that stopped it. An unreadable
/// frame (corruption, I/O) fails the whole pass with [`ReplayError::Log`],
/// after the projections that were still healthy have persisted the progress
/// they made up to it.
pub fn replay_all(
    log: &IntentLog,
    targets: &mut [&mut dyn ReplayTarget],
    persist_every: Option<u64>,
) -> Result<Vec<Result<u64, ReplayError>>, ReplayError> {
    let head_lsn = log.next_lsn();
    let mut lanes: Vec<Lane> = targets
        .iter()
        .map(|t| Lane::new(t.target_name(), t.target_checkpoint()))
        .collect();
    let start = lanes.iter().map(|l| l.resume_at).min();

    let span = tracing::info_span!(
        "projection.replay",
        projections = lanes.len(),
        start_lsn = start.map(|l| l.0),
        head_lsn = head_lsn.0,
        frames_read = tracing::field::Empty,
        payloads_opened = tracing::field::Empty,
    );
    let _guard = span.enter();

    let mut stats = ScanStats::default();
    let scan = match start {
        Some(start) if start < head_lsn => dispatch_frames(
            log,
            start,
            &mut lanes,
            targets,
            head_lsn,
            persist_every,
            &mut stats,
        ),
        _ => Ok(()),
    };

    for (lane, target) in lanes.iter_mut().zip(targets.iter_mut()) {
        if lane.failure.is_none() {
            lane.persist(&mut **target, head_lsn, "at end of replay");
        }
    }
    span.record("frames_read", stats.frames_read);
    span.record("payloads_opened", stats.payloads_opened);
    scan?;

    Ok(lanes
        .into_iter()
        .map(|lane| match lane.failure {
            Some(e) => Err(e),
            None => Ok(lane.applied),
        })
        .collect())
}

#[derive(Default)]
struct ScanStats {
    frames_read: u64,
    payloads_opened: u64,
}

fn dispatch_frames(
    log: &IntentLog,
    start: Lsn,
    lanes: &mut [Lane],
    targets: &mut [&mut dyn ReplayTarget],
    head_lsn: Lsn,
    persist_every: Option<u64>,
    stats: &mut ScanStats,
) -> Result<(), ReplayError> {
    let mut frames = log.iter_from(start)?;
    while let Some(frame) = frames.read_frame()? {
        stats.frames_read += 1;
        if !lanes.iter().any(|l| l.needs(frame.lsn)) {
            if lanes.iter().all(|l| l.failure.is_some()) {
                break;
            }
            continue;
        }
        stats.payloads_opened += 1;
        let record = IntentRecord {
            lsn: frame.lsn,
            payload: frames.take_payload()?,
        };
        for (lane, target) in lanes.iter_mut().zip(targets.iter_mut()) {
            if lane.needs(record.lsn) {
                lane.apply(&mut **target, &record, head_lsn, persist_every);
            }
        }
        frames.recycle(record.payload);
    }
    Ok(())
}

/// One projection's progress through a [`replay_all`] pass.
struct Lane {
    name: String,
    resume_at: Lsn,
    applied: u64,
    last_persisted_at: u64,
    failure: Option<ReplayError>,
}

impl Lane {
    fn new(name: &str, checkpoint: Option<Lsn>) -> Self {
        Self {
            name: name.to_string(),
            resume_at: checkpoint.map(Lsn::next).unwrap_or(Lsn::ZERO),
            applied: 0,
            last_persisted_at: 0,
            failure: None,
        }
    }

    fn needs(&self, lsn: Lsn) -> bool {
        self.failure.is_none() && lsn >= self.resume_at
    }

    fn apply(
        &mut self,
        target: &mut dyn ReplayTarget,
        record: &IntentRecord,
        head_lsn: Lsn,
        persist_every: Option<u64>,
    ) {
        let name = self.name.as_str();
        let timer = Timer::new(PROJECTION_APPLY_DURATION.with_label_values(&[name]));
        let apply_result = target.apply_record(record);
        drop(timer);
        if let Err(e) = apply_result {
            PROJECTION_APPLY_TOTAL
                .with_label_values(&[name, "error"])
                .inc();
            tracing::error!(
                projection = %name,
                lsn = record.lsn.0,
                error = %e,
                "projection apply failed during replay",
            );
            self.failure = Some(ReplayError::ProjectionFailed {
                projection: self.name.clone(),
                lsn: record.lsn,
                source: e,
            });
            return;
        }
        PROJECTION_APPLY_TOTAL
            .with_label_values(&[name, "ok"])
            .inc();
        PROJECTION_REPLAY_RECORDS_TOTAL
            .with_label_values(&[name])
            .inc();
        self.applied += 1;

        if let Some(n) = persist_every {
            if self.applied - self.last_persisted_at >= n
                && self.persist(target, head_lsn, "during replay")
            {
                self.last_persisted_at = self.applied;
            }
        }
    }

    /// Persist the projection's checkpoint. A failure is recorded against
    /// this lane and stops it; returns whether the persist succeeded.
    fn persist(&mut self, target: &mut dyn ReplayTarget, head_lsn: Lsn, when: &str) -> bool {
        let name = self.name.as_str();
        match target.persist_target_checkpoint() {
            Ok(()) => {
                CHECKPOINT_PERSIST_TOTAL
                    .with_label_values(&[name, "ok"])
                    .inc();
                publish_checkpoint_gauges(name, target.target_checkpoint(), head_lsn);
                true
            }
            Err(e) => {
                CHECKPOINT_PERSIST_TOTAL
                    .with_label_values(&[name, "error"])
                    .inc();
                let lsn = target.target_checkpoint().unwrap_or(Lsn::ZERO);
                tracing::error!(
                    projection = %name,
                    lsn = lsn.0,
                    error = %e,
                    "checkpoint persist failed {when}",
                );
                self.failure = Some(ReplayError::ProjectionFailed {
                    projection: self.name.clone(),
                    lsn,
                    source: e,
                });
                false
            }
        }
    }
}

/// Memory ids the log considers live at its head: created, updated or
/// anchored, and not forgotten since. The same set as
/// [`super::payload::live_memory_ids`], resumed from the snapshot an earlier
/// call left in `store` so only the frames appended since are read.
///
/// The snapshot is stamped with the [`super::FramePosition`] of the last
/// frame it covers and is used only while that exact frame (LSN and CRC) is
/// still at that offset; a truncated, restored or rewritten log fails the
/// check and the whole log is read. The memory id is taken from the encoded
/// payload's fixed prefix, so the memory snapshot inside a `Remember` or
/// `Update` is never decoded. The first frame of each kind is also decoded
/// in full and compared, and a disagreement switches the rest of the pass
/// to full decoding.
///
/// Persisting the refreshed snapshot is best effort: a failure is logged and
/// the next call reads further back.
pub fn live_memory_ids_resumable(
    log: &IntentLog,
    store: &Mutex<CheckpointStore>,
) -> Result<HashSet<String>, PayloadError> {
    let snapshot = store
        .lock()
        .get_side_data(INTENT_LOG_SIDE_DATA, LIVE_MEMORY_IDS_KEY)
        .and_then(LiveIdSnapshot::decode);
    let previous_stamp = snapshot.as_ref().map(|s| s.stamp);

    let (mut ids, mut frames, resume_at) = match snapshot {
        Some(snapshot) => match log.iter_at(snapshot.stamp)? {
            Some(frames) => (snapshot.ids, frames, snapshot.stamp.lsn.next()),
            None => {
                tracing::info!(
                    path = %log.path().display(),
                    stamp_lsn = snapshot.stamp.lsn.0,
                    "live-id snapshot no longer matches the intent log; reading the whole log",
                );
                (HashSet::new(), log.iter()?, Lsn::ZERO)
            }
        },
        None => (HashSet::new(), log.iter()?, Lsn::ZERO),
    };

    let mut prefix = PrefixReader::default();
    let mut last = None;
    while let Some(frame) = frames.read_frame()? {
        last = Some(frame.position());
        if frame.lsn < resume_at {
            continue;
        }
        let payload = frames.take_payload()?;
        let (forget, memory_id) = prefix.memory_ref(&payload)?;
        if forget {
            ids.remove(memory_id.as_ref());
        } else if !ids.contains(memory_id.as_ref()) {
            ids.insert(memory_id.into_owned());
        }
        frames.recycle(payload);
    }

    if let Some(stamp) = last.filter(|s| Some(*s) != previous_stamp) {
        let Some(encoded) = LiveIdSnapshot::encode(stamp, &ids) else {
            return Ok(ids);
        };
        let mut store = store.lock();
        let persisted = store
            .set_side_data_at(
                INTENT_LOG_SIDE_DATA,
                LIVE_MEMORY_IDS_KEY,
                stamp.lsn,
                &encoded,
            )
            .and_then(|()| store.sync_side_data());
        if let Err(e) = persisted {
            tracing::warn!(
                path = %log.path().display(),
                error = %e,
                ids = ids.len(),
                "could not persist the live-id snapshot; the next reconcile reads the whole log",
            );
        }
    }
    Ok(ids)
}

/// Variant indices of [`IntentPayload`] on the wire (declaration order).
const REMEMBER_TAG: u32 = 0;
const FORGET_TAG: u32 = 1;
const UPDATE_TAG: u32 = 2;
const ANCHOR_TAG: u32 = 3;
const SET_ANCHORED_TAG: u32 = 4;
const LAST_TAG: u32 = SET_ANCHORED_TAG;

fn variant_tag(payload: &IntentPayload) -> u32 {
    match payload {
        IntentPayload::Remember { .. } => REMEMBER_TAG,
        IntentPayload::Forget { .. } => FORGET_TAG,
        IntentPayload::Update { .. } => UPDATE_TAG,
        IntentPayload::Anchor { .. } => ANCHOR_TAG,
        IntentPayload::SetAnchored { .. } => SET_ANCHORED_TAG,
    }
}

/// Reads `(is_forget, memory_id)` off encoded payloads, from the prefix
/// every variant shares — `[variant u32][user_id: len u64, bytes][memory_id:
/// len u64, bytes]` under the payload's fixed-int bincode configuration —
/// cross-checked against a full decode for the first frame of each variant.
#[derive(Default)]
struct PrefixReader {
    checked: [bool; LAST_TAG as usize + 1],
    disabled: bool,
}

impl PrefixReader {
    fn memory_ref<'a>(
        &mut self,
        bytes: &'a [u8],
    ) -> Result<(bool, std::borrow::Cow<'a, str>), PayloadError> {
        if !self.disabled {
            if let Some((tag, id)) = peek_memory_ref(bytes) {
                let slot = tag as usize;
                if self.checked[slot] {
                    return Ok((tag == FORGET_TAG, id.into()));
                }
                let decoded = super::payload::decode(bytes)?;
                if decoded.memory_id() == id && variant_tag(&decoded) == tag {
                    self.checked[slot] = true;
                } else {
                    tracing::warn!(
                        variant = tag,
                        "intent payload prefix does not match its decoded form; decoding every frame",
                    );
                    self.disabled = true;
                }
                return Ok(full_memory_ref(decoded));
            }
        }
        Ok(full_memory_ref(super::payload::decode(bytes)?))
    }
}

fn full_memory_ref(payload: IntentPayload) -> (bool, std::borrow::Cow<'static, str>) {
    let forget = matches!(payload, IntentPayload::Forget { .. });
    (forget, payload.memory_id().to_owned().into())
}

/// `(variant, memory_id)` from an encoded payload's shared prefix, or `None`
/// when the bytes do not have that shape or name a variant this binary does
/// not know.
fn peek_memory_ref(bytes: &[u8]) -> Option<(u32, &str)> {
    fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
        Some(u64::from_le_bytes(
            bytes.get(at..at.checked_add(8)?)?.try_into().ok()?,
        ))
    }
    let tag = u32::from_le_bytes(bytes.get(0..4)?.try_into().ok()?);
    if tag > LAST_TAG {
        return None;
    }
    let user_len = usize::try_from(read_u64(bytes, 4)?).ok()?;
    let id_len_at = 12usize.checked_add(user_len)?;
    let id_len = usize::try_from(read_u64(bytes, id_len_at)?).ok()?;
    let id_at = id_len_at.checked_add(8)?;
    let id = bytes.get(id_at..id_at.checked_add(id_len)?)?;
    Some((tag, std::str::from_utf8(id).ok()?))
}

const LIVE_ID_SNAPSHOT_VERSION: u8 = 1;

/// The live-id set as of one frame: `[version u8][stamp][count u32]` then
/// each id as `[len u16][utf-8 bytes]`, little-endian.
struct LiveIdSnapshot {
    stamp: FramePosition,
    ids: HashSet<String>,
}

impl LiveIdSnapshot {
    /// `None` when an id is too long for the format or the set too large
    /// to keep as side data; the caller then keeps no snapshot.
    fn encode(stamp: FramePosition, ids: &HashSet<String>) -> Option<Vec<u8>> {
        let body: usize = ids.iter().map(|id| 2 + id.len()).sum();
        let total = 1 + FRAME_POSITION_BYTES + 4 + body;
        if total > MAX_SIDE_DATA_BYTES || ids.iter().any(|id| id.len() > u16::MAX as usize) {
            return None;
        }
        let mut out = Vec::with_capacity(total);
        out.push(LIVE_ID_SNAPSHOT_VERSION);
        out.extend_from_slice(&stamp.to_bytes());
        out.extend_from_slice(&u32::try_from(ids.len()).ok()?.to_le_bytes());
        for id in ids {
            out.extend_from_slice(&(id.len() as u16).to_le_bytes());
            out.extend_from_slice(id.as_bytes());
        }
        Some(out)
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let (&version, rest) = bytes.split_first()?;
        if version != LIVE_ID_SNAPSHOT_VERSION {
            return None;
        }
        let stamp = FramePosition::from_bytes(rest.get(..FRAME_POSITION_BYTES)?)?;
        let rest = &rest[FRAME_POSITION_BYTES..];
        let count = u32::from_le_bytes(rest.get(..4)?.try_into().ok()?) as usize;
        let mut rest = &rest[4..];
        let mut ids = HashSet::with_capacity(count.min(rest.len() / 2));
        for _ in 0..count {
            let len = u16::from_le_bytes(rest.get(..2)?.try_into().ok()?) as usize;
            let id = std::str::from_utf8(rest.get(2..2 + len)?).ok()?;
            ids.insert(id.to_owned());
            rest = &rest[2 + len..];
        }
        rest.is_empty().then_some(Self { stamp, ids })
    }
}

/// Publish (or refresh) the `veld_projection_checkpoint_lsn` and
/// `veld_projection_lag_records` gauges for one projection. Centralised so
/// every callsite produces the same shape of metric labels.
fn publish_checkpoint_gauges(projection: &str, checkpoint: Option<Lsn>, head_lsn: Lsn) {
    let checkpoint_lsn = checkpoint.map(|l| l.0).unwrap_or(0);
    PROJECTION_CHECKPOINT_LSN
        .with_label_values(&[projection])
        .set(checkpoint_lsn as i64);
    // Lag = head_lsn - (checkpoint_lsn + 1) records still to apply, clamped
    // to zero. When checkpoint is None we have not applied anything yet, so
    // lag is the full head value.
    let next_to_apply = match checkpoint {
        Some(l) => l.0.saturating_add(1),
        None => 0,
    };
    let lag = head_lsn.0.saturating_sub(next_to_apply);
    PROJECTION_LAG_RECORDS
        .with_label_values(&[projection])
        .set(lag as i64);
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
                return Err(FlakyError(format!(
                    "seeded failure at lsn {:?}",
                    record.lsn
                )));
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
        p.push(format!(
            "veld-intent-log-projection-{name}-{pid}-{stamp}.log"
        ));
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

    /// Minimal observability smoke test: after one successful append + one
    /// successful apply (via replay), the relevant Prometheus counters have
    /// advanced and the checkpoint gauge matches the applied LSN.
    #[test]
    fn metrics_advance_after_one_append_and_one_apply() {
        use crate::metrics::{
            INTENT_LOG_APPEND_TOTAL, INTENT_LOG_SYNC_TOTAL, PROJECTION_APPLY_TOTAL,
            PROJECTION_CHECKPOINT_LSN, PROJECTION_REPLAY_RECORDS_TOTAL,
        };

        // Use a unique projection name so the counters are not polluted by
        // other tests running in the same process.
        let projection_name = format!("metric-smoke-{}", std::process::id());

        let append_before = INTENT_LOG_APPEND_TOTAL.with_label_values(&["ok"]).get();
        let sync_before = INTENT_LOG_SYNC_TOTAL.with_label_values(&["ok"]).get();
        let apply_before = PROJECTION_APPLY_TOTAL
            .with_label_values(&[projection_name.as_str(), "ok"])
            .get();
        let replay_before = PROJECTION_REPLAY_RECORDS_TOTAL
            .with_label_values(&[projection_name.as_str()])
            .get();

        let path = tmp_log_path("metrics_smoke");
        {
            let mut log = IntentLog::open(&path).unwrap();
            log.append(b"observe-me").unwrap();
            log.sync().unwrap();
        }

        let log = IntentLog::open(&path).unwrap();
        let mut proj = VecProjection::new(&projection_name);
        let applied = replay(&log, &mut proj, None).unwrap();
        assert_eq!(applied, 1);

        let append_after = INTENT_LOG_APPEND_TOTAL.with_label_values(&["ok"]).get();
        let sync_after = INTENT_LOG_SYNC_TOTAL.with_label_values(&["ok"]).get();
        let apply_after = PROJECTION_APPLY_TOTAL
            .with_label_values(&[projection_name.as_str(), "ok"])
            .get();
        let replay_after = PROJECTION_REPLAY_RECORDS_TOTAL
            .with_label_values(&[projection_name.as_str()])
            .get();

        assert!(
            append_after >= append_before + 1,
            "intent_log_append_total{{result=ok}} did not advance: {} -> {}",
            append_before,
            append_after
        );
        assert!(
            sync_after >= sync_before + 1,
            "intent_log_sync_total{{result=ok}} did not advance: {} -> {}",
            sync_before,
            sync_after
        );
        assert!(
            apply_after >= apply_before + 1,
            "projection_apply_total{{result=ok}} did not advance: {} -> {}",
            apply_before,
            apply_after
        );
        assert!(
            replay_after >= replay_before + 1,
            "projection_replay_records_total did not advance: {} -> {}",
            replay_before,
            replay_after
        );

        // Checkpoint gauge should match the only applied LSN (0).
        let cp_lsn = PROJECTION_CHECKPOINT_LSN
            .with_label_values(&[projection_name.as_str()])
            .get();
        assert_eq!(
            cp_lsn, 0,
            "checkpoint gauge should reflect last applied lsn"
        );

        let _ = std::fs::remove_file(&path);
    }

    fn numbered_log(
        name: &str,
        count: u8,
        cryptor: Option<crate::keystore::SecretCryptor>,
    ) -> (PathBuf, IntentLog) {
        let path = tmp_log_path(name);
        let mut log = IntentLog::open_with_cryptor(&path, cryptor).unwrap();
        for i in 0..count {
            log.append(&[i]).unwrap();
        }
        log.sync().unwrap();
        (path, log)
    }

    fn key() -> Option<crate::keystore::SecretCryptor> {
        Some(crate::keystore::SecretCryptor::new(
            zeroize::Zeroizing::new([5u8; 32]),
        ))
    }

    fn at_checkpoint(name: &str, checkpoint: Option<Lsn>) -> VecProjection {
        let mut proj = VecProjection::new(name);
        proj.checkpoint = checkpoint;
        proj.persisted_checkpoint = checkpoint;
        proj
    }

    #[test]
    fn replay_all_applies_each_frame_once_to_each_projection_below_it() {
        let (path, log) = numbered_log("all_once", 8, None);
        let mut fresh = at_checkpoint("fresh", None);
        let mut mid = at_checkpoint("mid", Some(Lsn(2)));
        let mut current = at_checkpoint("current", Some(Lsn(7)));
        let results = {
            let mut targets: [&mut dyn ReplayTarget; 3] = [&mut fresh, &mut mid, &mut current];
            replay_all(&log, &mut targets, Some(3)).unwrap()
        };
        let applied: Vec<u64> = results.into_iter().map(Result::unwrap).collect();
        assert_eq!(applied, vec![8, 5, 0]);

        let lsns = |p: &VecProjection| p.applied.iter().map(|(l, _)| l.0).collect::<Vec<_>>();
        assert_eq!(lsns(&fresh), (0..8).collect::<Vec<_>>());
        assert_eq!(lsns(&mid), (3..8).collect::<Vec<_>>());
        assert!(current.applied.is_empty());
        assert_eq!(mid.applied[0].1, vec![3u8]);
        for p in [&fresh, &mid, &current] {
            assert_eq!(p.persisted_checkpoint, Some(Lsn(7)), "{}", p.name);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn replay_all_keeps_a_failed_projection_from_blocking_the_others() {
        let (path, log) = numbered_log("all_failure", 6, None);
        let mut flaky = FlakyProjection {
            name: "flaky".to_string(),
            applied: Vec::new(),
            checkpoint: None,
            fail_once_at: Some(Lsn(2)),
        };
        let mut steady = at_checkpoint("steady", Some(Lsn(0)));
        let results = {
            let mut targets: [&mut dyn ReplayTarget; 2] = [&mut flaky, &mut steady];
            replay_all(&log, &mut targets, None).unwrap()
        };
        match &results[0] {
            Err(ReplayError::ProjectionFailed {
                projection, lsn, ..
            }) => {
                assert_eq!(projection, "flaky");
                assert_eq!(*lsn, Lsn(2));
            }
            other => panic!("expected the flaky projection to fail, got {other:?}"),
        }
        assert_eq!(*results[1].as_ref().unwrap(), 5);
        assert_eq!(flaky.applied, vec![Lsn(0), Lsn(1)]);
        assert_eq!(flaky.checkpoint, Some(Lsn(1)));
        assert_eq!(steady.persisted_checkpoint, Some(Lsn(5)));

        // The next pass retries the failed record for the stalled projection
        // only; the caught-up one is not re-fed.
        let results = {
            let mut targets: [&mut dyn ReplayTarget; 2] = [&mut flaky, &mut steady];
            replay_all(&log, &mut targets, None).unwrap()
        };
        assert_eq!(*results[0].as_ref().unwrap(), 4);
        assert_eq!(*results[1].as_ref().unwrap(), 0);
        assert_eq!(flaky.applied, (0..6).map(Lsn).collect::<Vec<_>>());
        assert_eq!(steady.applied.len(), 5);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn frames_no_projection_needs_are_never_decrypted() {
        let (path, log) = numbered_log("no_decrypt", 10, key());
        let mut behind = at_checkpoint("behind", Some(Lsn(7)));
        let mut ahead = at_checkpoint("ahead", Some(Lsn(8)));
        let before = super::super::payload_decrypts_on_this_thread();
        let results = {
            let mut targets: [&mut dyn ReplayTarget; 2] = [&mut behind, &mut ahead];
            replay_all(&log, &mut targets, None).unwrap()
        };
        assert_eq!(
            super::super::payload_decrypts_on_this_thread() - before,
            2,
            "only LSNs 8 and 9 are opened"
        );
        assert_eq!(*results[0].as_ref().unwrap(), 2);
        assert_eq!(*results[1].as_ref().unwrap(), 1);
        assert_eq!(behind.applied[0], (Lsn(8), vec![8u8]));
        assert_eq!(ahead.applied, vec![(Lsn(9), vec![9u8])]);

        // Fully caught up: nothing is read or opened at all.
        let before = super::super::payload_decrypts_on_this_thread();
        let mut targets: [&mut dyn ReplayTarget; 2] = [&mut behind, &mut ahead];
        replay_all(&log, &mut targets, None).unwrap();
        assert_eq!(super::super::payload_decrypts_on_this_thread(), before);
        let _ = std::fs::remove_file(&path);
    }

    fn payload(kind: u32, memory_id: &str) -> IntentPayload {
        let user_id = "tenant-with-a-longer-name".to_string();
        let memory_id = memory_id.to_string();
        let schema_version = Some(super::super::payload::CURRENT_PAYLOAD_SCHEMA_VERSION);
        match kind {
            REMEMBER_TAG => IntentPayload::Remember {
                user_id,
                memory_id,
                memory_bincode: vec![9; 64],
                schema_version,
            },
            FORGET_TAG => IntentPayload::Forget {
                user_id,
                memory_id,
                schema_version,
            },
            UPDATE_TAG => IntentPayload::Update {
                user_id,
                memory_id,
                memory_bincode: vec![7; 16],
                schema_version,
            },
            ANCHOR_TAG => IntentPayload::Anchor {
                user_id,
                memory_id,
                importance: 0.9,
                schema_version,
            },
            _ => IntentPayload::SetAnchored {
                user_id,
                memory_id,
                anchored: false,
                importance: 0.4,
                schema_version,
            },
        }
    }

    #[test]
    fn the_payload_prefix_names_the_same_variant_and_memory_as_a_full_decode() {
        for kind in [
            REMEMBER_TAG,
            FORGET_TAG,
            UPDATE_TAG,
            ANCHOR_TAG,
            SET_ANCHORED_TAG,
        ] {
            let original = payload(kind, "4a1f0c3e-0000-4000-8000-00000000abcd");
            let bytes = super::super::payload::encode(&original).unwrap();
            let (tag, id) = peek_memory_ref(&bytes).expect("prefix parses");
            assert_eq!(tag, variant_tag(&original));
            assert_eq!(id, original.memory_id());
            let mut reader = PrefixReader::default();
            let (forget, id) = reader.memory_ref(&bytes).unwrap();
            assert_eq!(forget, kind == FORGET_TAG);
            assert_eq!(id, original.memory_id());
            assert!(!reader.disabled);
        }
        assert!(peek_memory_ref(&[0, 0, 0]).is_none());
        assert!(peek_memory_ref(&[9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).is_none());
    }

    #[test]
    fn live_ids_resume_from_the_snapshot_and_fall_back_when_it_goes_stale() {
        use super::super::payload::append;
        let path = tmp_log_path("live_ids");
        let ckpt_path = tmp_log_path("live_ids_ckpt");
        let store = Mutex::new(CheckpointStore::open(&ckpt_path).unwrap());
        let mut log = IntentLog::open_with_cryptor(&path, key()).unwrap();
        for (kind, id) in [
            (REMEMBER_TAG, "m0"),
            (REMEMBER_TAG, "m1"),
            (FORGET_TAG, "m1"),
            (UPDATE_TAG, "m2"),
            (ANCHOR_TAG, "m3"),
            (SET_ANCHORED_TAG, "m3"),
        ] {
            append(&mut log, &payload(kind, id)).unwrap();
        }
        log.sync().unwrap();
        let set = |ids: &[&str]| ids.iter().map(|s| s.to_string()).collect::<HashSet<_>>();

        let first = live_memory_ids_resumable(&log, &store).unwrap();
        assert_eq!(first, set(&["m0", "m2", "m3"]));
        assert_eq!(first, super::super::payload::live_memory_ids(&log).unwrap());

        append(&mut log, &payload(FORGET_TAG, "m0")).unwrap();
        append(&mut log, &payload(REMEMBER_TAG, "m4")).unwrap();
        log.sync().unwrap();
        let before = super::super::payload_decrypts_on_this_thread();
        let second = live_memory_ids_resumable(&log, &store).unwrap();
        assert_eq!(second, set(&["m2", "m3", "m4"]));
        assert_eq!(
            super::super::payload_decrypts_on_this_thread() - before,
            2,
            "only the two frames after the snapshot are opened"
        );

        // A reopened store resumes from the persisted snapshot too.
        drop(store);
        let store = Mutex::new(CheckpointStore::open(&ckpt_path).unwrap());
        let before = super::super::payload_decrypts_on_this_thread();
        assert_eq!(live_memory_ids_resumable(&log, &store).unwrap(), second);
        assert_eq!(super::super::payload_decrypts_on_this_thread(), before);

        // Truncating below the snapshot's frame invalidates it.
        log.truncate_to_lsn(Lsn(1)).unwrap();
        assert_eq!(
            live_memory_ids_resumable(&log, &store).unwrap(),
            set(&["m0", "m1"])
        );
        let side = store.lock().side_data_path().to_path_buf();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&ckpt_path);
        let _ = std::fs::remove_file(&side);
    }
}
