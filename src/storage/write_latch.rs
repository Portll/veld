//! Detection, bounded recovery and honest reporting for a RocksDB write latch.
//!
//! After a background IO error RocksDB stops accepting writes. For the
//! `kNoSpace` subcode it never clears that by itself: the retryable-IO
//! auto-resume path is bypassed for NoSpace (`db/error_handler.cc:446`) and the
//! only remaining route needs an `SstFileManager` attached to the options, for
//! which the C API has no setter (`db/error_handler.cc:541`). `DB::Resume()` is
//! not exposed by rust-rocksdb 0.24 either. So a hard latch is cleared by
//! restarting the process and by nothing else, and the job here is to notice,
//! try the part that *is* recoverable, and say so where a reader will look.
//!
//! Measured 2026-10-11: 11,088 consecutive `No space left on device` write
//! failures against a 221 MB store on a volume with 253 GiB free, cleared only
//! by a restart, while `/health` reported `healthy` throughout.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::Serialize;

/// sysexits(3) `EX_TEMPFAIL` — "temporary failure, retry invited", which is the
/// thing a supervisor needs to hear to restart us into a reopened store.
pub const EXIT_WRITE_LATCHED: i32 = 75;

const DEFAULT_MIN_FREE_BYTES: u64 = 256 * 1024 * 1024;
const DEFAULT_MAX_ATTEMPTS: u32 = 3;
const DEFAULT_COOLDOWN_SECS: u64 = 30;

fn parse_u64(raw: Result<String, std::env::VarError>, default: u64) -> u64 {
    raw.ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

// The variable names are written out at each call site rather than passed to one
// helper, so `docs/generators/src/bin/gen-config-ref.rs` — which matches literal
// `env::var("VELD_…")` — lists them. A knob the configuration reference cannot
// see is a knob nobody finds.

/// Read at call time, never cached at module load: a cached read defeats every
/// test and every operator who sets the variable afterwards.
pub fn min_free_bytes() -> u64 {
    parse_u64(
        std::env::var("VELD_WRITE_LATCH_MIN_FREE_BYTES"),
        DEFAULT_MIN_FREE_BYTES,
    )
}

pub fn max_attempts() -> u32 {
    parse_u64(
        std::env::var("VELD_WRITE_LATCH_MAX_ATTEMPTS"),
        u64::from(DEFAULT_MAX_ATTEMPTS),
    ) as u32
}

pub fn cooldown() -> Duration {
    Duration::from_secs(parse_u64(
        std::env::var("VELD_WRITE_LATCH_COOLDOWN_SECS"),
        DEFAULT_COOLDOWN_SECS,
    ))
}

fn restart_on_hard_latch() -> bool {
    matches!(
        std::env::var("VELD_WRITE_LATCH_RESTART").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// What the latch knows, in the shape `/health` publishes.
#[derive(Debug, Clone, Serialize)]
pub struct LatchSnapshot {
    /// The verbatim error chain that latched it.
    pub reason: String,
    /// Store path the failing write targeted.
    pub path: String,
    pub since: DateTime<Utc>,
    pub failed_writes: u64,
    pub recovery_attempts: u32,
    /// A flush+retry has already failed; only a restart clears it.
    pub hard: bool,
}

#[derive(Debug)]
struct Latched {
    reason: String,
    path: String,
    since: DateTime<Utc>,
    failed_writes: u64,
    recovery_attempts: u32,
    hard: bool,
    hard_logged: bool,
    last_probe: Option<Instant>,
}

/// One store's write latch. Construct one per test; production reads
/// [`global()`].
#[derive(Debug, Default)]
pub struct WriteLatch {
    /// Mirrors `state.is_some()` so the success path costs one relaxed load
    /// rather than a mutex acquisition per write.
    latched: AtomicBool,
    state: Mutex<Option<Latched>>,
}

impl WriteLatch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a failed write. Returns true when the error is a NoSpace latch —
    /// newly latching or continuing one — which is the caller's cue to attempt
    /// recovery.
    pub fn note_write_error(&self, path: &Path, err: &anyhow::Error) -> bool {
        if !chain_is_enospc(err) {
            return false;
        }
        let reason = format!("{err:#}");
        let mut state = self.state.lock();
        match state.as_mut() {
            Some(latched) => {
                // `since` is deliberately untouched: one latch has one start time.
                latched.failed_writes += 1;
            }
            None => {
                *state = Some(Latched {
                    reason: reason.clone(),
                    path: path.display().to_string(),
                    since: Utc::now(),
                    failed_writes: 1,
                    recovery_attempts: 0,
                    hard: false,
                    hard_logged: false,
                    last_probe: None,
                });
                self.latched.store(true, Ordering::Release);
                tracing::warn!(
                    path = %path.display(),
                    reason = %reason,
                    "storage writes latched: RocksDB reported no space left on device"
                );
            }
        }
        true
    }

    /// A write got through. This is the unlatch.
    pub fn note_write_ok(&self) {
        if !self.latched.load(Ordering::Relaxed) {
            return;
        }
        let mut state = self.state.lock();
        if let Some(latched) = state.take() {
            self.latched.store(false, Ordering::Release);
            tracing::info!(
                path = %latched.path,
                failed_writes = latched.failed_writes,
                recovery_attempts = latched.recovery_attempts,
                was_hard = latched.hard,
                "storage write latch cleared by a successful write"
            );
        }
    }

    /// A flush+retry has failed. Past here only a restart clears the store.
    pub fn mark_hard(&self) {
        let mut state = self.state.lock();
        if let Some(latched) = state.as_mut() {
            latched.hard = true;
        }
    }

    /// True exactly once per latch, so the unrecoverable notice is logged once
    /// instead of once per request against a latched store.
    pub fn claim_hard_notice(&self) -> bool {
        let mut state = self.state.lock();
        match state.as_mut() {
            Some(latched) if !latched.hard_logged => {
                latched.hard_logged = true;
                true
            }
            _ => false,
        }
    }

    pub fn snapshot(&self) -> Option<LatchSnapshot> {
        self.state.lock().as_ref().map(|latched| LatchSnapshot {
            reason: latched.reason.clone(),
            path: latched.path.clone(),
            since: latched.since,
            failed_writes: latched.failed_writes,
            recovery_attempts: latched.recovery_attempts,
            hard: latched.hard,
        })
    }

    pub fn is_latched(&self) -> bool {
        self.latched.load(Ordering::Relaxed)
    }

    pub fn is_hard(&self) -> bool {
        self.state
            .lock()
            .as_ref()
            .is_some_and(|latched| latched.hard)
    }

    /// Whether a recovery probe may run now, stamping the cooldown as it
    /// answers. Separate from the attempt budget so that declining to retry
    /// into a still-full disk does not spend one, which would leave the store
    /// latched for good once space came back.
    pub fn probe_due(&self) -> bool {
        let mut state = self.state.lock();
        let Some(latched) = state.as_mut() else {
            return false;
        };
        let now = Instant::now();
        if let Some(previous) = latched.last_probe {
            if now.duration_since(previous) < cooldown() {
                return false;
            }
        }
        latched.last_probe = Some(now);
        true
    }

    /// Whether the bounded flush+retry budget has anything left.
    pub fn attempts_left(&self) -> bool {
        let limit = max_attempts();
        self.state
            .lock()
            .as_ref()
            .is_some_and(|latched| latched.recovery_attempts < limit)
    }

    /// Spend one flush+retry attempt.
    pub fn note_recovery_attempt(&self) {
        let mut state = self.state.lock();
        if let Some(latched) = state.as_mut() {
            latched.recovery_attempts += 1;
        }
    }
}

/// The process-wide latch the write paths and `/health` share.
pub fn global() -> &'static WriteLatch {
    static GLOBAL: OnceLock<WriteLatch> = OnceLock::new();
    GLOBAL.get_or_init(WriteLatch::new)
}

/// ENOSPC is the one subcode RocksDB's error handler refuses to auto-recover
/// without an SstFileManager, which the C API cannot register — see
/// `error_handler.cc:446,541`. Every other write error either retries itself or
/// means something a restart will not fix.
pub fn is_enospc(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("no space left on device") || lowered.contains("enospc")
}

/// The same question over a whole `anyhow` chain. Production messages arrive
/// wrapped (`Failed to put memory <uuid> in RocksDB: …`), so `to_string()` on
/// the outer error alone sees only the wrapper.
pub fn chain_is_enospc(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| is_enospc(&cause.to_string()))
}

/// Free bytes on the filesystem holding `path`, by longest mount-point prefix.
/// `None` means unmeasurable, which is neither zero nor plenty.
pub fn available_bytes(path: &Path) -> Option<u64> {
    // A failing store path may not exist yet; the nearest existing ancestor is
    // on the same filesystem for every case this gate is about.
    let probe = path
        .ancestors()
        .find_map(|candidate| candidate.canonicalize().ok())
        .unwrap_or_else(|| path.to_path_buf());

    let disks = sysinfo::Disks::new_with_refreshed_list();
    let mut best: Option<(usize, u64)> = None;
    for disk in disks.list() {
        let mount = disk.mount_point();
        if !probe.starts_with(mount) {
            continue;
        }
        let depth = mount.components().count();
        match best {
            Some((best_depth, _)) if best_depth >= depth => {}
            _ => best = Some((depth, disk.available_space())),
        }
    }
    best.map(|(_, free)| free)
}

/// Run `retry` once if `err` latched the store and the filesystem looks
/// writable again; otherwise return the error that got us here.
///
/// RocksDB maps some write-path errors to `kSoftError`, which clears on a
/// successful retry, so a flush+retry genuinely unlatches part of the
/// population. For the rest this marks the latch hard and says what a reader
/// must do about it.
pub fn recover_write<T>(
    latch: &WriteLatch,
    db: &rocksdb::DB,
    path: &Path,
    err: anyhow::Error,
    retry: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if !latch.note_write_error(path, &err) {
        return Err(err);
    }
    if !latch.probe_due() {
        return Err(err);
    }

    let min_free = min_free_bytes();
    let space_back = available_bytes(path).map(|free| free >= min_free);

    if latch.is_hard() {
        if space_back == Some(true) {
            maybe_restart(path);
        }
        return Err(err);
    }

    match space_back {
        Some(false) => {
            tracing::warn!(
                path = %path.display(),
                min_free_bytes = min_free,
                "storage writes latched and the filesystem is still full — not retrying"
            );
            return Err(err);
        }
        Some(true) => {}
        // Reported, never substituted for a reading: one bounded flush+retry is
        // the cheapest way to learn whether the store can take a write.
        None => tracing::warn!(
            path = %path.display(),
            "storage writes latched and free space could not be measured — attempting recovery anyway"
        ),
    }

    if !latch.attempts_left() {
        return Err(err);
    }
    latch.note_recovery_attempt();

    if let Err(flush_err) = db.flush() {
        latch.mark_hard();
        report_hard(latch, path, &format!("flush failed: {flush_err}"));
        if space_back == Some(true) {
            maybe_restart(path);
        }
        return Err(err);
    }

    match retry() {
        Ok(value) => {
            latch.note_write_ok();
            tracing::info!(
                path = %path.display(),
                "storage write latch was soft: flush and retry succeeded"
            );
            Ok(value)
        }
        Err(retry_err) => {
            latch.mark_hard();
            report_hard(latch, path, &format!("{retry_err:#}"));
            if space_back == Some(true) {
                maybe_restart(path);
            }
            Err(retry_err)
        }
    }
}

fn report_hard(latch: &WriteLatch, path: &Path, detail: &str) {
    if !latch.claim_hard_notice() {
        return;
    }
    tracing::error!(
        path = %path.display(),
        detail = %detail,
        exit_code_on_restart = EXIT_WRITE_LATCHED,
        "storage writes are hard-latched — restart this process to recover. \
         RocksDB cannot clear a NoSpace background error in-process: its auto-resume \
         path excludes that subcode (error_handler.cc:446) and the only alternative \
         needs an SstFileManager the C API has no setter for (error_handler.cc:541)"
    );
}

/// Opt-in, default off: exiting is a behaviour change an operator chooses.
fn maybe_restart(path: &Path) {
    if !restart_on_hard_latch() {
        return;
    }
    tracing::error!(
        path = %path.display(),
        exit_code = EXIT_WRITE_LATCHED,
        "exiting for a supervised restart: storage writes are hard-latched and the \
         filesystem has space again"
    );
    std::process::exit(EXIT_WRITE_LATCHED);
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{anyhow, Context};
    use std::path::PathBuf;

    /// Taken verbatim from the captured production log, with the operator's path
    /// generalised. A classifier checked only against a string we invented has
    /// no floor.
    const PRODUCTION_MESSAGE: &str =
        "IO error: No space left on device: While open a file for appending: /store/000585.log";

    fn store_path() -> PathBuf {
        PathBuf::from("/store")
    }

    #[test]
    fn is_enospc_matches_the_verbatim_production_message() {
        assert!(is_enospc(PRODUCTION_MESSAGE));
    }

    #[test]
    fn is_enospc_matches_both_production_wrappers() {
        assert!(is_enospc(&format!(
            "Failed to put memory 00000000-0000-0000-0000-000000000000 in RocksDB: {PRODUCTION_MESSAGE}"
        )));
        assert!(is_enospc(&format!(
            "Failed to flush memory storage: {PRODUCTION_MESSAGE}"
        )));
    }

    #[test]
    fn is_enospc_rejects_other_write_errors() {
        assert!(!is_enospc("IO error: Permission denied"));
        assert!(!is_enospc("Corruption: block checksum mismatch"));
        assert!(!is_enospc("NotFound: "));
        assert!(!is_enospc(""));
    }

    #[test]
    fn chain_is_enospc_walks_past_the_context_wrapper() {
        let wrapped = Err::<(), _>(anyhow!("{PRODUCTION_MESSAGE}"))
            .context("Failed to put memory 00000000-0000-0000-0000-000000000000 in RocksDB")
            .unwrap_err();
        // The outermost message alone says nothing about space.
        assert!(!is_enospc(&wrapped.to_string()));
        assert!(chain_is_enospc(&wrapped));
    }

    #[test]
    fn note_write_error_latches_and_the_snapshot_carries_the_evidence() {
        let latch = WriteLatch::new();
        assert!(!latch.is_latched());
        assert!(latch.snapshot().is_none());

        let err = anyhow!("{PRODUCTION_MESSAGE}");
        assert!(latch.note_write_error(&store_path(), &err));

        assert!(latch.is_latched());
        let snapshot = latch.snapshot().expect("latched");
        assert!(snapshot.reason.contains("No space left on device"));
        assert_eq!(snapshot.path, "/store");
        assert_eq!(snapshot.failed_writes, 1);
        assert_eq!(snapshot.recovery_attempts, 0);
        assert!(!snapshot.hard);
        assert!(snapshot.since <= Utc::now());
    }

    #[test]
    fn a_non_enospc_error_does_not_latch() {
        let latch = WriteLatch::new();
        let err = anyhow!("Corruption: block checksum mismatch");
        assert!(!latch.note_write_error(&store_path(), &err));
        assert!(!latch.is_latched());
        assert!(latch.snapshot().is_none());
    }

    #[test]
    fn note_write_ok_clears_the_latch() {
        let latch = WriteLatch::new();
        latch.note_write_error(&store_path(), &anyhow!("{PRODUCTION_MESSAGE}"));
        assert!(latch.is_latched());

        latch.note_write_ok();
        assert!(!latch.is_latched());
        assert!(latch.snapshot().is_none());
    }

    #[test]
    fn failed_writes_accumulate_and_since_does_not_move() {
        let latch = WriteLatch::new();
        let err = anyhow!("{PRODUCTION_MESSAGE}");
        latch.note_write_error(&store_path(), &err);
        let first = latch.snapshot().expect("latched");

        for _ in 0..4 {
            assert!(latch.note_write_error(&store_path(), &err));
        }
        let later = latch.snapshot().expect("still latched");

        assert_eq!(later.failed_writes, 5);
        assert_eq!(later.since, first.since);
    }

    #[test]
    fn mark_hard_sets_hard_and_survives_further_errors() {
        let latch = WriteLatch::new();
        latch.note_write_error(&store_path(), &anyhow!("{PRODUCTION_MESSAGE}"));
        assert!(!latch.is_hard());

        latch.mark_hard();
        assert!(latch.is_hard());

        latch.note_write_error(&store_path(), &anyhow!("{PRODUCTION_MESSAGE}"));
        assert!(latch.is_hard());
        assert!(latch.snapshot().expect("latched").hard);
    }

    #[test]
    fn the_hard_notice_is_claimable_once_per_latch() {
        let latch = WriteLatch::new();
        latch.note_write_error(&store_path(), &anyhow!("{PRODUCTION_MESSAGE}"));
        latch.mark_hard();

        assert!(latch.claim_hard_notice());
        assert!(!latch.claim_hard_notice());

        // A new latch gets its own notice.
        latch.note_write_ok();
        latch.note_write_error(&store_path(), &anyhow!("{PRODUCTION_MESSAGE}"));
        assert!(latch.claim_hard_notice());
    }

    /// Asserted against the defaults rather than an env override: the env is
    /// process-global and these tests run in parallel with everything else.
    #[test]
    fn the_recovery_budget_is_bounded() {
        let latch = WriteLatch::new();
        latch.note_write_error(&store_path(), &anyhow!("{PRODUCTION_MESSAGE}"));

        let limit = max_attempts();
        for _ in 0..limit {
            assert!(latch.attempts_left());
            latch.note_recovery_attempt();
        }
        assert!(!latch.attempts_left());
        assert_eq!(latch.snapshot().expect("latched").recovery_attempts, limit);
    }

    #[test]
    fn a_second_probe_inside_the_cooldown_is_declined() {
        let latch = WriteLatch::new();
        // Nothing latched: nothing to probe for.
        assert!(!latch.probe_due());

        latch.note_write_error(&store_path(), &anyhow!("{PRODUCTION_MESSAGE}"));
        assert!(latch.probe_due());
        assert!(
            !latch.probe_due(),
            "a latched store under load must not pay for a filesystem probe per write"
        );
    }

    #[test]
    fn available_bytes_reads_the_filesystem_holding_the_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Not an assertion about capacity — only that a real path resolves to a
        // real mount. An unmeasurable reading is its own case in recover_write.
        match available_bytes(dir.path()) {
            Some(free) => assert!(free > 0, "a writable tempdir reported zero free bytes"),
            None => {
                eprintln!("free space unmeasurable on this host — recover_write warns and retries")
            }
        }
    }

    // ---- Second witness: a REAL rocksdb::Error through the same plumbing ----
    //
    // The classifier's own failure mode is a hand-built string that flatters it,
    // so these drive an error RocksDB actually produced. A real ENOSPC
    // `rocksdb::Error` cannot be minted here (`rocksdb::Error::new` is private
    // and there is no `/dev/full` on macOS), so the false-negative direction is
    // asserted over a real error carrying the production text in its chain.

    fn a_real_rocksdb_error() -> rocksdb::Error {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("db");
        {
            let mut opts = rocksdb::Options::default();
            opts.create_if_missing(true);
            let db = rocksdb::DB::open(&opts, &path).expect("open");
            db.put(b"k", b"v").expect("seed write");
        }
        let read_only = rocksdb::DB::open_for_read_only(&rocksdb::Options::default(), &path, false)
            .expect("reopen read-only");
        read_only
            .put(b"k", b"v2")
            .expect_err("a read-only DB must refuse a write")
    }

    #[test]
    fn false_positive_direction_a_real_rocksdb_error_is_not_enospc() {
        let raw = a_real_rocksdb_error();
        let message = raw.to_string();
        // Floor: an empty message would pass the negative assertion for free.
        assert!(
            !message.is_empty(),
            "rocksdb returned an empty error message, so this witness proves nothing"
        );

        assert!(!is_enospc(&message), "misclassified: {message}");

        let wrapped = anyhow::Error::new(raw)
            .context("Failed to put memory 00000000-0000-0000-0000-000000000000 in RocksDB");
        assert!(
            !chain_is_enospc(&wrapped),
            "the production wrapper must not latch on its own: {wrapped:#}"
        );

        let latch = WriteLatch::new();
        assert!(!latch.note_write_error(&store_path(), &wrapped));
        assert!(!latch.is_latched());
    }

    #[test]
    fn false_negative_direction_a_real_error_chain_carrying_enospc_does_latch() {
        let raw = a_real_rocksdb_error();
        let wrapped = anyhow::Error::new(raw)
            .context(PRODUCTION_MESSAGE)
            .context("Failed to put memory 00000000-0000-0000-0000-000000000000 in RocksDB");

        assert!(
            chain_is_enospc(&wrapped),
            "the classifier missed ENOSPC in a real error chain: {wrapped:#}"
        );

        let latch = WriteLatch::new();
        assert!(latch.note_write_error(&store_path(), &wrapped));
        assert!(latch.is_latched());
    }

    /// `recover_write` end to end over a real DB and a real refusal: it must
    /// return the caller's error untouched and leave nothing latched. Driving
    /// the whole body matters because the classifier being right does not make
    /// the surrounding decisions right.
    #[test]
    fn recover_write_passes_a_non_enospc_error_straight_through() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("db");
        {
            let mut opts = rocksdb::Options::default();
            opts.create_if_missing(true);
            let db = rocksdb::DB::open(&opts, &path).expect("open");
            db.put(b"k", b"v").expect("seed write");
        }
        let read_only = rocksdb::DB::open_for_read_only(&rocksdb::Options::default(), &path, false)
            .expect("reopen read-only");

        let latch = WriteLatch::new();
        let mut retried = false;
        let err = anyhow::Error::new(
            read_only
                .put(b"k", b"v2")
                .expect_err("a read-only DB must refuse a write"),
        )
        .context("Failed to put memory 00000000-0000-0000-0000-000000000000 in RocksDB");
        let original = format!("{err:#}");

        let outcome: Result<()> = recover_write(&latch, &read_only, &path, err, || {
            retried = true;
            Ok(())
        });

        let returned = outcome.expect_err("the caller's error must survive");
        assert_eq!(format!("{returned:#}"), original);
        assert!(!retried, "a non-NoSpace error must not be retried");
        assert!(!latch.is_latched());
        assert!(latch.snapshot().is_none());
    }
}
