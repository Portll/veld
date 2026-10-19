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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest, Sha256};

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

/// The only condition that latches a store here, as a fixed token rather than
/// the RocksDB message — see [`LatchSnapshot`] for why the message cannot ship.
pub const LATCH_CONDITION: &str = "no_space_left_on_device";

/// What `/health` publishes about one latched store.
///
/// **Carries no store path and no error chain, deliberately.** `/health` and
/// `/health/ready` are on the unauthenticated probe router (`PROBE_PATHS`,
/// `handlers/router.rs`), and every per-user store directory is
/// `base.join(user_id)` (`path_guard.rs:3`). A path would hand an anonymous
/// caller the host filesystem layout and a tenant id, and an array of them
/// would be tenant enumeration — the same disclosure the public router's own
/// contract forbids by the `?user_id=` route. The error chain is worse than the
/// path: RocksDB's message embeds the absolute file being appended to, and the
/// wrapper embeds the failing memory's UUID.
///
/// `store` is a stable opaque id for the path instead, so an operator can tell
/// two latched stores apart and line a response up with the log. The verbatim
/// path and chain go to the WARN and ERROR log lines, which are not publicly
/// readable.
#[derive(Debug, Clone, Serialize)]
pub struct LatchSnapshot {
    /// Store path the failing write targeted: for the log line, never the body.
    #[serde(skip)]
    pub path: String,
    /// The verbatim error chain that latched it: for the log line, never the body.
    #[serde(skip)]
    pub reason: String,
    /// Truncated SHA-256 of the store path. See [`store_id`].
    pub store: String,
    /// Always [`LATCH_CONDITION`]; named so the body says what is wrong.
    pub condition: &'static str,
    pub since: DateTime<Utc>,
    pub failed_writes: u64,
    pub recovery_attempts: u32,
    /// A flush+retry has already failed; only a restart clears it.
    pub hard: bool,
}

/// Stable opaque id for a store path: the first 16 hex characters of its
/// SHA-256. Stable across responses and restarts so it can be correlated with
/// the log line that does carry the path, and not a tenant id.
pub fn store_id(path: &Path) -> String {
    let digest = Sha256::digest(path.as_os_str().as_encoded_bytes());
    hex::encode(&digest[..8])
}

#[derive(Debug)]
struct Latched {
    reason: String,
    since: DateTime<Utc>,
    failed_writes: u64,
    recovery_attempts: u32,
    hard: bool,
    hard_logged: bool,
    last_probe: Option<Instant>,
}

impl Latched {
    /// The publishable projection. One place builds it, so a field added to the
    /// record does not reach a public body by being forgotten about.
    fn publish(&self, path: &Path) -> LatchSnapshot {
        LatchSnapshot {
            path: path.display().to_string(),
            reason: self.reason.clone(),
            store: store_id(path),
            condition: LATCH_CONDITION,
            since: self.since,
            failed_writes: self.failed_writes,
            recovery_attempts: self.recovery_attempts,
            hard: self.hard,
        }
    }
}

/// Write latches for every store in this process, keyed by store path.
///
/// Both repos run one RocksDB per tenant (`base.join(user_id)`), and RocksDB's
/// own latch is per-database. A process-wide flag would therefore clear tenant
/// B's latch on a successful write to tenant A — the store would be reported
/// usable on evidence about a different store, which is the exact defect this
/// module exists to remove.
///
/// Construct one per test; production reads [`global()`].
#[derive(Debug, Default)]
pub struct WriteLatch {
    /// Mirrors `!stores.is_empty()` so the success path costs one relaxed load
    /// rather than a mutex acquisition per write.
    any_latched: AtomicBool,
    stores: Mutex<HashMap<PathBuf, Latched>>,
}

impl WriteLatch {
    pub fn new() -> Self {
        Self::default()
    }

    /// The map key. The store path as the write sites already hold it — no
    /// canonicalisation, which would be a syscall on the hot path and could
    /// fail for the very store that is failing.
    fn key(path: &Path) -> PathBuf {
        path.to_path_buf()
    }

    /// Record a failed write against `path`. Returns true when the error is a
    /// NoSpace latch — newly latching or continuing one — which is the caller's
    /// cue to attempt recovery.
    pub fn note_write_error(&self, path: &Path, err: &anyhow::Error) -> bool {
        if !chain_is_enospc(err) {
            return false;
        }
        let reason = format!("{err:#}");
        let mut stores = self.stores.lock();
        match stores.get_mut(&Self::key(path)) {
            Some(latched) => {
                // `since` is deliberately untouched: one latch has one start time.
                latched.failed_writes += 1;
            }
            None => {
                stores.insert(
                    Self::key(path),
                    Latched {
                        reason: reason.clone(),
                        since: Utc::now(),
                        failed_writes: 1,
                        recovery_attempts: 0,
                        hard: false,
                        hard_logged: false,
                        last_probe: None,
                    },
                );
                self.any_latched.store(true, Ordering::Release);
                tracing::warn!(
                    path = %path.display(),
                    reason = %reason,
                    "storage writes latched: RocksDB reported no space left on device"
                );
            }
        }
        true
    }

    /// A write got through to `path`. This is the unlatch, and it clears only
    /// this store — a write that landed elsewhere says nothing about this one.
    pub fn note_write_ok(&self, path: &Path) {
        if !self.any_latched.load(Ordering::Relaxed) {
            return;
        }
        let mut stores = self.stores.lock();
        if let Some(latched) = stores.remove(&Self::key(path)) {
            self.any_latched
                .store(!stores.is_empty(), Ordering::Release);
            tracing::info!(
                path = %path.display(),
                failed_writes = latched.failed_writes,
                recovery_attempts = latched.recovery_attempts,
                was_hard = latched.hard,
                stores_still_latched = stores.len(),
                "storage write latch cleared by a successful write"
            );
        }
    }

    /// A flush+retry has failed for `path`. Past here only a restart clears it.
    pub fn mark_hard(&self, path: &Path) {
        if let Some(latched) = self.stores.lock().get_mut(&Self::key(path)) {
            latched.hard = true;
        }
    }

    /// True exactly once per latch, so the unrecoverable notice is logged once
    /// instead of once per request against a latched store.
    pub fn claim_hard_notice(&self, path: &Path) -> bool {
        match self.stores.lock().get_mut(&Self::key(path)) {
            Some(latched) if !latched.hard_logged => {
                latched.hard_logged = true;
                true
            }
            _ => false,
        }
    }

    /// Every latched store, in the publishable shape. Ordered by `store` so the
    /// response is deterministic across calls.
    pub fn snapshot(&self) -> Vec<LatchSnapshot> {
        let stores = self.stores.lock();
        let mut out: Vec<LatchSnapshot> = stores
            .iter()
            .map(|(path, latched)| latched.publish(path))
            .collect();
        out.sort_by(|a, b| a.store.cmp(&b.store));
        out
    }

    /// This store's latch, if it has one.
    pub fn snapshot_at(&self, path: &Path) -> Option<LatchSnapshot> {
        self.stores
            .lock()
            .get(&Self::key(path))
            .map(|latched| latched.publish(path))
    }

    /// The verbatim chain that latched this store. For log lines only — never
    /// for a response body; see [`LatchSnapshot`].
    pub fn reason_at(&self, path: &Path) -> Option<String> {
        self.stores
            .lock()
            .get(&Self::key(path))
            .map(|latched| latched.reason.clone())
    }

    /// Whether ANY store is latched.
    pub fn is_latched(&self) -> bool {
        self.any_latched.load(Ordering::Relaxed)
    }

    pub fn is_latched_at(&self, path: &Path) -> bool {
        self.stores.lock().contains_key(&Self::key(path))
    }

    /// Whether ANY store is hard-latched.
    pub fn is_hard(&self) -> bool {
        self.stores.lock().values().any(|latched| latched.hard)
    }

    pub fn is_hard_at(&self, path: &Path) -> bool {
        self.stores
            .lock()
            .get(&Self::key(path))
            .is_some_and(|latched| latched.hard)
    }

    /// How many stores are hard-latched. Reported on `/health/ready`, because
    /// "one store of twelve" and "all twelve" need different responses from a
    /// human and the status code cannot tell them apart.
    pub fn hard_count(&self) -> usize {
        self.stores
            .lock()
            .values()
            .filter(|latched| latched.hard)
            .count()
    }

    /// Whether a recovery probe may run now for `path`, stamping the cooldown as
    /// it answers. Separate from the attempt budget so that declining to retry
    /// into a still-full disk does not spend one, which would leave the store
    /// latched for good once space came back.
    pub fn probe_due(&self, path: &Path) -> bool {
        let mut stores = self.stores.lock();
        let Some(latched) = stores.get_mut(&Self::key(path)) else {
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

    /// Whether this store's bounded flush+retry budget has anything left.
    pub fn attempts_left(&self, path: &Path) -> bool {
        let limit = max_attempts();
        self.stores
            .lock()
            .get(&Self::key(path))
            .is_some_and(|latched| latched.recovery_attempts < limit)
    }

    /// Spend one flush+retry attempt against `path`.
    pub fn note_recovery_attempt(&self, path: &Path) {
        if let Some(latched) = self.stores.lock().get_mut(&Self::key(path)) {
            latched.recovery_attempts += 1;
        }
    }
}

/// The per-process registry the write paths and `/health` share. Entries are
/// added when a store latches and removed when it clears, so the map cannot
/// grow past the tenant count and needs no eviction policy.
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

/// Run one write against `db` through the [`global`] latch: a success clears
/// this store's latch, a refusal goes to [`recover_write`].
///
/// Keyed by `db.path()` and nothing else. Several adapters write to one
/// database — the memory store, its batch writer and its key-value views share
/// a tenant's `storage` directory — and a key chosen per call site let a write
/// through one of them leave the latch another had raised standing for good.
///
/// `op` is called at most twice, so it must be replayable.
#[cfg(feature = "rocksdb")]
pub fn guard<T>(db: &rocksdb::DB, op: impl Fn() -> Result<T>) -> Result<T> {
    let path = db.path();
    let attempt = || {
        #[cfg(test)]
        injected::refuse_if_armed(path)?;
        op()
    };
    let latch = global();
    match attempt() {
        Ok(value) => {
            latch.note_write_ok(path);
            Ok(value)
        }
        Err(err) => recover_write(latch, db, path, err, attempt),
    }
}

/// [`guard`] for a batch already built. `DB::write_opt` consumes its batch, so
/// each attempt writes a copy and the retry still has the original to copy.
#[cfg(feature = "rocksdb")]
pub fn guard_batch(
    db: &rocksdb::DB,
    batch: &rocksdb::WriteBatch,
    opts: &rocksdb::WriteOptions,
) -> Result<()> {
    guard(db, || {
        db.write_opt(rocksdb::WriteBatch::from_data(batch.data()), opts)
            .map_err(anyhow::Error::new)
    })
}

/// Per-thread NoSpace refusals for [`guard`], so a test can drive a real write
/// site into the latch. RocksDB will not mint an ENOSPC error on demand (see the
/// witness tests below), and a thread-local keeps one test's refusals out of
/// every other test's writes.
#[cfg(all(test, feature = "rocksdb"))]
pub(crate) mod injected {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    /// The production log line [`super::is_enospc`] is pinned to.
    const NO_SPACE: &str =
        "IO error: No space left on device: While open a file for appending: /store/000585.log";

    thread_local! {
        /// Refusals left per store path; `None` refuses until disarmed.
        static ARMED: RefCell<HashMap<PathBuf, Option<u32>>> = RefCell::new(HashMap::new());
    }

    /// Refuse every write to `path` on this thread until [`disarm`].
    pub(crate) fn arm(path: &Path) {
        ARMED.with(|armed| armed.borrow_mut().insert(path.to_path_buf(), None));
    }

    /// Refuse the next `count` writes to `path` on this thread.
    pub(crate) fn arm_times(path: &Path, count: u32) {
        if count > 0 {
            ARMED.with(|armed| armed.borrow_mut().insert(path.to_path_buf(), Some(count)));
        }
    }

    pub(crate) fn disarm(path: &Path) {
        ARMED.with(|armed| armed.borrow_mut().remove(path));
    }

    pub(super) fn refuse_if_armed(path: &Path) -> anyhow::Result<()> {
        let refuse = ARMED.with(|armed| {
            let mut armed = armed.borrow_mut();
            match armed.get_mut(path) {
                None => false,
                Some(None) => true,
                Some(Some(left)) => {
                    *left -= 1;
                    if *left == 0 {
                        armed.remove(path);
                    }
                    true
                }
            }
        });
        if refuse {
            Err(anyhow::anyhow!("{NO_SPACE}"))
        } else {
            Ok(())
        }
    }
}

/// Run `retry` once if `err` latched the store and the filesystem looks
/// writable again; otherwise return the error that got us here.
///
/// RocksDB maps some write-path errors to `kSoftError`, which clears on a
/// successful retry, so a flush+retry genuinely unlatches part of the
/// population. For the rest this marks the latch hard and says what a reader
/// must do about it.
#[cfg(feature = "rocksdb")]
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
    if !latch.probe_due(path) {
        return Err(err);
    }

    let min_free = min_free_bytes();
    let space_back = available_bytes(path).map(|free| free >= min_free);

    if latch.is_hard_at(path) {
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

    if !latch.attempts_left(path) {
        return Err(err);
    }
    latch.note_recovery_attempt(path);

    if let Err(flush_err) = db.flush() {
        latch.mark_hard(path);
        report_hard(latch, path, &format!("flush failed: {flush_err}"));
        if space_back == Some(true) {
            maybe_restart(path);
        }
        return Err(err);
    }

    match retry() {
        Ok(value) => {
            latch.note_write_ok(path);
            tracing::info!(
                path = %path.display(),
                "storage write latch was soft: flush and retry succeeded"
            );
            Ok(value)
        }
        Err(retry_err) => {
            latch.mark_hard(path);
            report_hard(latch, path, &format!("{retry_err:#}"));
            if space_back == Some(true) {
                maybe_restart(path);
            }
            Err(retry_err)
        }
    }
}

fn report_hard(latch: &WriteLatch, path: &Path, detail: &str) {
    // `reason_at` before `claim_hard_notice`: the notice is claimable once, and
    // a clear racing in between would leave the one ERROR line without a cause.
    let latched_by = latch.reason_at(path).unwrap_or_default();
    if !latch.claim_hard_notice(path) {
        return;
    }
    // The path and the verbatim chain live here and only here — the `/health`
    // body cannot carry either (see `LatchSnapshot`), so this line has to be
    // enough on its own for an operator reading only errors.
    tracing::error!(
        path = %path.display(),
        store = %store_id(path),
        latched_by = %latched_by,
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
        assert!(latch.snapshot().is_empty());

        let err = anyhow!("{PRODUCTION_MESSAGE}");
        assert!(latch.note_write_error(&store_path(), &err));

        assert!(latch.is_latched());
        assert!(latch.is_latched_at(&store_path()));
        let snapshot = latch.snapshot_at(&store_path()).expect("latched");
        assert_eq!(snapshot.store, store_id(&store_path()));
        assert_eq!(snapshot.condition, LATCH_CONDITION);
        assert_eq!(snapshot.failed_writes, 1);
        assert_eq!(snapshot.recovery_attempts, 0);
        assert!(!snapshot.hard);
        assert!(snapshot.since <= Utc::now());

        // The chain is kept for the log line, not for the response body.
        assert!(latch
            .reason_at(&store_path())
            .expect("latched")
            .contains("No space left on device"));
    }

    #[test]
    fn a_non_enospc_error_does_not_latch() {
        let latch = WriteLatch::new();
        let err = anyhow!("Corruption: block checksum mismatch");
        assert!(!latch.note_write_error(&store_path(), &err));
        assert!(!latch.is_latched());
        assert!(latch.snapshot().is_empty());
    }

    #[test]
    fn note_write_ok_clears_the_latch() {
        let latch = WriteLatch::new();
        latch.note_write_error(&store_path(), &anyhow!("{PRODUCTION_MESSAGE}"));
        assert!(latch.is_latched());

        latch.note_write_ok(&store_path());
        assert!(!latch.is_latched());
        assert!(latch.snapshot().is_empty());
    }

    /// The round-1 defect: the latch was process-global, so this cleared store
    /// A on evidence that only ever concerned store B.
    #[test]
    fn a_write_landing_on_another_store_does_not_clear_this_one() {
        let latch = WriteLatch::new();
        let a = Path::new("/var/lib/veld/users/tenant-a/storage");
        let b = Path::new("/var/lib/veld/users/tenant-b/storage");

        latch.note_write_error(a, &anyhow!("{PRODUCTION_MESSAGE}"));
        latch.note_write_ok(b);

        assert!(
            latch.is_latched_at(a),
            "tenant A's store is still unwritable and nothing has measured it"
        );
        assert!(latch.is_latched());
        assert_eq!(latch.snapshot().len(), 1);
    }

    #[test]
    fn two_stores_latch_independently_and_clear_independently() {
        let latch = WriteLatch::new();
        let a = Path::new("/var/lib/veld/users/tenant-a/storage");
        let b = Path::new("/var/lib/veld/users/tenant-b/storage");

        latch.note_write_error(a, &anyhow!("{PRODUCTION_MESSAGE}"));
        latch.note_write_error(b, &anyhow!("{PRODUCTION_MESSAGE}"));
        let both = latch.snapshot();
        assert_eq!(both.len(), 2);
        let ids: Vec<&str> = both.iter().map(|s| s.store.as_str()).collect();
        assert!(ids.contains(&store_id(a).as_str()));
        assert!(ids.contains(&store_id(b).as_str()));

        latch.note_write_ok(a);
        assert!(!latch.is_latched_at(a));
        assert!(latch.is_latched_at(b));
        assert!(latch.is_latched(), "B is still latched, so something is");
        assert_eq!(latch.snapshot().len(), 1);

        latch.note_write_ok(b);
        assert!(!latch.is_latched());
        assert!(latch.snapshot().is_empty());
    }

    #[test]
    fn hard_is_per_store_and_counted() {
        let latch = WriteLatch::new();
        let a = Path::new("/var/lib/veld/users/tenant-a/storage");
        let b = Path::new("/var/lib/veld/users/tenant-b/storage");

        latch.note_write_error(a, &anyhow!("{PRODUCTION_MESSAGE}"));
        latch.note_write_error(b, &anyhow!("{PRODUCTION_MESSAGE}"));
        latch.mark_hard(b);

        assert!(!latch.is_hard_at(a));
        assert!(latch.is_hard_at(b));
        assert!(latch.is_hard(), "any store hard-latched makes this true");
        assert_eq!(latch.hard_count(), 1);

        latch.mark_hard(a);
        assert_eq!(latch.hard_count(), 2);
    }

    #[test]
    fn the_map_holds_only_latched_stores() {
        let latch = WriteLatch::new();
        let a = Path::new("/var/lib/veld/users/tenant-a/storage");

        // A successful write to a store that never latched inserts nothing.
        latch.note_write_ok(a);
        assert!(latch.snapshot().is_empty());

        latch.note_write_error(a, &anyhow!("{PRODUCTION_MESSAGE}"));
        assert_eq!(latch.snapshot().len(), 1);
        latch.note_write_ok(a);
        assert!(latch.snapshot().is_empty());
    }

    #[test]
    fn failed_writes_accumulate_and_since_does_not_move() {
        let latch = WriteLatch::new();
        let err = anyhow!("{PRODUCTION_MESSAGE}");
        latch.note_write_error(&store_path(), &err);
        let first = latch.snapshot_at(&store_path()).expect("latched");

        for _ in 0..4 {
            assert!(latch.note_write_error(&store_path(), &err));
        }
        let later = latch.snapshot_at(&store_path()).expect("still latched");

        assert_eq!(later.failed_writes, 5);
        assert_eq!(later.since, first.since);
    }

    #[test]
    fn mark_hard_sets_hard_and_survives_further_errors() {
        let latch = WriteLatch::new();
        latch.note_write_error(&store_path(), &anyhow!("{PRODUCTION_MESSAGE}"));
        assert!(!latch.is_hard());

        latch.mark_hard(&store_path());
        assert!(latch.is_hard());

        latch.note_write_error(&store_path(), &anyhow!("{PRODUCTION_MESSAGE}"));
        assert!(latch.is_hard());
        assert!(latch.snapshot_at(&store_path()).expect("latched").hard);
    }

    #[test]
    fn the_hard_notice_is_claimable_once_per_latch() {
        let latch = WriteLatch::new();
        latch.note_write_error(&store_path(), &anyhow!("{PRODUCTION_MESSAGE}"));
        latch.mark_hard(&store_path());

        assert!(latch.claim_hard_notice(&store_path()));
        assert!(!latch.claim_hard_notice(&store_path()));

        // A new latch gets its own notice.
        latch.note_write_ok(&store_path());
        latch.note_write_error(&store_path(), &anyhow!("{PRODUCTION_MESSAGE}"));
        assert!(latch.claim_hard_notice(&store_path()));
    }

    /// Asserted against the defaults rather than an env override: the env is
    /// process-global and these tests run in parallel with everything else.
    #[test]
    fn the_recovery_budget_is_bounded_per_store() {
        let latch = WriteLatch::new();
        let a = Path::new("/var/lib/veld/users/tenant-a/storage");
        let b = Path::new("/var/lib/veld/users/tenant-b/storage");
        latch.note_write_error(a, &anyhow!("{PRODUCTION_MESSAGE}"));
        latch.note_write_error(b, &anyhow!("{PRODUCTION_MESSAGE}"));

        let limit = max_attempts();
        for _ in 0..limit {
            assert!(latch.attempts_left(a));
            latch.note_recovery_attempt(a);
        }
        assert!(!latch.attempts_left(a));
        assert_eq!(
            latch.snapshot_at(a).expect("latched").recovery_attempts,
            limit
        );
        assert!(
            latch.attempts_left(b),
            "A spending its budget must not spend B's"
        );
    }

    #[test]
    fn a_second_probe_inside_the_cooldown_is_declined_per_store() {
        let latch = WriteLatch::new();
        let a = Path::new("/var/lib/veld/users/tenant-a/storage");
        let b = Path::new("/var/lib/veld/users/tenant-b/storage");

        // Nothing latched: nothing to probe for.
        assert!(!latch.probe_due(a));

        latch.note_write_error(a, &anyhow!("{PRODUCTION_MESSAGE}"));
        assert!(latch.probe_due(a));
        assert!(
            !latch.probe_due(a),
            "a latched store under load must not pay for a filesystem probe per write"
        );

        latch.note_write_error(b, &anyhow!("{PRODUCTION_MESSAGE}"));
        assert!(
            latch.probe_due(b),
            "A's cooldown must not suppress B's first probe"
        );
    }

    #[test]
    fn store_id_is_stable_distinct_and_leaks_no_path() {
        let a = Path::new("/var/lib/veld/users/tenant-7f3a9c2b/storage");
        let b = Path::new("/var/lib/veld/users/tenant-other/storage");

        assert_eq!(store_id(a), store_id(a), "must be stable across calls");
        assert_ne!(store_id(a), store_id(b));
        assert_eq!(store_id(a).len(), 16);
        assert!(store_id(a).chars().all(|c| c.is_ascii_hexdigit()));
        assert!(!store_id(a).contains('/'));
        assert!(!store_id(a).contains("tenant-7f3a9c2b"));
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

    #[cfg(feature = "rocksdb")]
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

    #[cfg(feature = "rocksdb")]
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

    #[cfg(feature = "rocksdb")]
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
    #[cfg(feature = "rocksdb")]
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
        assert!(latch.snapshot().is_empty());
    }

    /// The retry writes a copy rebuilt from the batch's serialized form, which
    /// names its families by id; the copy has to land where the original would.
    #[cfg(feature = "rocksdb")]
    #[test]
    fn guard_batch_retries_a_refused_batch_into_the_same_family() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let db = rocksdb::DB::open_cf(&opts, dir.path(), ["default", "rows"]).expect("open");
        let rows = db.cf_handle("rows").expect("family");
        let mut batch = rocksdb::WriteBatch::default();
        batch.put_cf(rows, b"k", b"v");

        injected::arm_times(db.path(), 1);
        let outcome = guard_batch(&db, &batch, &rocksdb::WriteOptions::default());
        injected::disarm(db.path());

        // recover_write retries only past the free-space floor, or when free
        // space cannot be measured.
        let retried = available_bytes(db.path()).is_none_or(|free| free >= min_free_bytes());
        if retried {
            outcome.expect("the retry writes the copy");
            assert_eq!(
                db.get_cf(rows, b"k").expect("read").as_deref(),
                Some(&b"v"[..])
            );
            assert!(
                db.get(b"k").expect("read").is_none(),
                "the copy landed outside its family"
            );
            assert!(!global().is_latched_at(db.path()));
        } else {
            let err = outcome.expect_err("a full disk is not retried");
            assert!(chain_is_enospc(&err));
            assert!(global().is_latched_at(db.path()));
            global().note_write_ok(db.path());
        }
    }
}
