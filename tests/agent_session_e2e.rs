//! End-to-end verification that the AgentSession writer-side wiring reads
//! the `.veld-agent-session.<pid>` marker file the SessionStart hook drops
//! and stamps every persisted memory with the resulting session identity.
//!
//! The detection chain (file -> env var -> parent process -> None) is unit
//! tested in `src/memory/session_detect.rs`; this file exercises the chain
//! through the same writer call sites the running server uses, so any
//! regression in the *wiring* (missing context plumbing, lost AgentSession
//! between `Experience` construction and RocksDB persistence) is caught
//! independently of the detection helper itself.
//!
//! Run with: `cargo test --test agent_session_e2e`

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex as StdMutex;

use tempfile::TempDir;

use veld::memory::session_detect;
use veld::memory::types::{Experience, ExperienceType};
use veld::memory::{MemoryConfig, MemoryId, MemorySystem};

/// Global lock for tests that mutate process-wide state (`cwd`,
/// `VELD_AGENT_ID`). Rust's harness runs integration tests in parallel; cwd
/// and env are shared across all threads, so without serialization one test
/// would silently clobber another's setup.
static PROCESS_STATE_GUARD: StdMutex<()> = StdMutex::new(());

const SESSION_FILE_PREFIX: &str = ".veld-agent-session.";

/// Test fixture: snapshot+restore cwd and `VELD_AGENT_ID`, switch cwd to a
/// fresh tempdir, invalidate the session cache. Drop restores everything.
///
/// The fixture also exposes `pid_file_path()` so callers can write the
/// `.veld-agent-session.<pid>` marker into the same dir the detection helper
/// is about to read.
struct Fixture {
    _lock: std::sync::MutexGuard<'static, ()>,
    original_cwd: PathBuf,
    original_agent_id: Option<std::ffi::OsString>,
    tempdir: TempDir,
}

impl Fixture {
    fn new() -> Self {
        // A poisoned guard means a prior test panicked. The state is whatever
        // that test left, which we are about to reset anyway — either inner
        // value is a valid guard.
        let lock = PROCESS_STATE_GUARD
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let original_cwd = env::current_dir().expect("cwd readable");
        let original_agent_id = env::var_os("VELD_AGENT_ID");
        // SAFETY: tests run single-threaded under PROCESS_STATE_GUARD; no
        // other code in this process is mutating VELD_AGENT_ID concurrently.
        env::remove_var("VELD_AGENT_ID");

        let tempdir = TempDir::new().expect("create tempdir");
        env::set_current_dir(tempdir.path()).expect("chdir to tempdir");
        session_detect::invalidate_session_cache();

        Self {
            _lock: lock,
            original_cwd,
            original_agent_id,
            tempdir,
        }
    }

    fn pid_file_path(&self) -> PathBuf {
        self.tempdir
            .path()
            .join(format!("{SESSION_FILE_PREFIX}{}", std::process::id()))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // Restore the original VELD_AGENT_ID (or remove if unset).
        if let Some(val) = self.original_agent_id.take() {
            env::set_var("VELD_AGENT_ID", val);
        } else {
            env::remove_var("VELD_AGENT_ID");
        }
        let _ = env::set_current_dir(&self.original_cwd);
        session_detect::invalidate_session_cache();
    }
}

/// Initialise an empty git repo in `dir` so the session-detect helper's
/// `git rev-parse --abbrev-ref HEAD` probe has a checkout to talk to.
/// Returns `true` on success; the caller can decide whether the test
/// requires git or treats it as best-effort.
fn init_git_repo(dir: &Path) -> bool {
    let init = Command::new("git")
        .args(["init", "--initial-branch=test-branch"])
        .current_dir(dir)
        .output();
    let init_ok = matches!(init, Ok(out) if out.status.success());
    if !init_ok {
        // `--initial-branch` is git >= 2.28; fall back to a default init
        // plus an explicit symbolic-ref switch for older binaries.
        let init = Command::new("git").args(["init"]).current_dir(dir).output();
        if !matches!(init, Ok(out) if out.status.success()) {
            return false;
        }
        let _ = Command::new("git")
            .args(["symbolic-ref", "HEAD", "refs/heads/test-branch"])
            .current_dir(dir)
            .output();
    }
    true
}

/// Spin up a fresh `MemorySystem` rooted in `path`.
fn fresh_memory_system(path: PathBuf) -> MemorySystem {
    let config = MemoryConfig {
        storage_path: path,
        collective_store_dir: None,
        working_memory_size: 16,
        session_memory_size_mb: 16,
        max_heap_per_user_mb: 64,
        auto_compress: false,
        compression_age_days: 7,
        importance_threshold: 0.0,
    };
    MemorySystem::new(config, None).expect("create memory system")
}

// =============================================================================
// Tests
// =============================================================================

/// Smoke test: a `.veld-agent-session.<pid>` marker in cwd is read end-to-end
/// — from helper through writer through RocksDB readback. The test initialises
/// a fresh git repo inside the tempdir so the branch probe has a checkout to
/// resolve; the marker file then drives `agent_id`, and RocksDB readback
/// confirms the AgentSession survived persistence.
#[test]
fn marker_file_round_trips_into_stored_memory() {
    let fx = Fixture::new();

    // Initialise a git repo in the tempdir so the session-detect helper's
    // `git rev-parse --abbrev-ref HEAD` probe returns a real branch name.
    // The branch is intentionally distinctive (`test-branch`) so the
    // assertion below cannot accidentally pass on a stale cached value.
    let git_ready = init_git_repo(fx.tempdir.path());
    assert!(
        git_ready,
        "git init failed — the test environment needs a working `git` binary on PATH"
    );

    // Write the SessionStart hook's payload.
    fs::write(
        fx.pid_file_path(),
        r#"{"agent_id":"Claude","started_at":"2026-05-28T00:00:00Z","pid":1,"binary":"claude-code"}"#,
    )
    .expect("write marker file");

    // Force the next detect_session() to consult the file we just wrote.
    session_detect::invalidate_session_cache();

    // Build the same RichContext the writer-side helpers construct, then
    // hand it to MemorySystem::remember — the exact path the HTTP handler
    // (and ingest/upsert/zenoh) use.
    let context = Some(session_detect::rich_context_with_session());
    let experience = Experience {
        experience_type: ExperienceType::Observation,
        content: "marker-file-round-trips".to_string(),
        context,
        ..Default::default()
    };

    // Storage path lives inside the tempdir so the rocksdb lives and dies
    // with the test.
    let storage = fx.tempdir.path().join("veld-storage");
    fs::create_dir_all(&storage).expect("mkdir storage");
    let memory = fresh_memory_system(storage);

    let id: MemoryId = memory
        .remember(experience, None)
        .expect("remember succeeded");

    let stored = memory.get_memory(&id).expect("memory readable");
    let ctx = stored
        .experience
        .context
        .as_ref()
        .expect("stored memory carries a RichContext");

    assert_eq!(
        ctx.session.agent_id.as_deref(),
        Some("Claude"),
        "marker file should populate agent_id end-to-end, got {:?}",
        ctx.session.agent_id
    );

    // Git probe should have resolved `test-branch` via `git init
    // --initial-branch=test-branch` (or the symbolic-ref fallback). Pinning
    // the exact value rules out the helper silently returning a stale
    // cached branch from an outer process.
    assert_eq!(
        ctx.session.branch.as_deref(),
        Some("test-branch"),
        "branch detection should resolve `test-branch`, got {:?}",
        ctx.session.branch
    );

    // The marker file is cleaned up automatically when `fx` drops the
    // tempdir, but be explicit so the assertion above is unambiguous about
    // what made the round-trip work.
    let _ = fs::remove_file(fx.pid_file_path());
}

/// Cross-platform check: `worktree_path` is stored as POSIX (forward-slash)
/// regardless of host OS, so memories exported across platforms agree on
/// the canonical path shape.
#[test]
fn worktree_path_is_posix_normalised() {
    let _fx = Fixture::new();
    session_detect::invalidate_session_cache();

    let ctx = session_detect::rich_context_with_session();
    let path = ctx
        .session
        .worktree_path
        .as_ref()
        .expect("worktree_path populated");
    let s = path.to_str().expect("UTF-8 worktree path");
    assert!(
        !s.contains('\\'),
        "worktree_path should be POSIX-normalised, contained a backslash: {s:?}"
    );
}

/// Negative control: when the marker file is absent and `VELD_AGENT_ID` is
/// unset, the writer still produces a `RichContext` (so memories never lose
/// the facet entirely), but `agent_id` is whatever parent-process detection
/// returns — `None` under `cargo test`, where the parent is `cargo` or a
/// shell. This pins the contract for tests not running under a known agent.
#[test]
fn writer_emits_context_even_without_marker() {
    let _fx = Fixture::new();
    session_detect::invalidate_session_cache();

    let context = session_detect::rich_context_with_session();
    // The session struct itself is always present; only the fields inside
    // it are optional. Same shape as the marker-file case above, but with
    // an `agent_id` we can't pin (parent-process detection might match in
    // some CI environments — assertion would be flaky).
    assert!(context.session.started_at.is_some());
    assert!(context.session.worktree_path.is_some());
}
