//! Auto-detection of [`AgentSession`] identity for the current process.
//!
//! Veld memories carry an [`AgentSession`] facet — *who ran from where, on
//! which branch*. The facet exists on every record, but populating it has to
//! happen on the writer side at request time: the storage layer doesn't know
//! which agent (Claude / Copilot / Cursor / …) is actually driving Veld, nor
//! which sibling worktree the agent is running from.
//!
//! This module is the single source of truth for that detection. It runs once
//! per process (results are cached) and is consulted by every `RichContext`
//! construction site, so any memory written from a given Veld process shares
//! the same session identity.
//!
//! # Detection chain — `agent_id`
//!
//! The chat-brand identity is resolved in priority order, with anything
//! outside the [`AGENT_ID_WHITELIST`] rejected so a hostile environment cannot
//! masquerade as an arbitrary chat client:
//!
//! 1. **Sentinel file.** A `.veld-agent-session.<pid>` file in the current
//!    working directory (`<pid>` = the calling process's PID). The file is
//!    JSON; only the `agent_id` field is consulted today. Brand-only
//!    detection — anything else in the file is currently ignored. This is the
//!    hook channel: a wrapper script (the Claude Code launcher, for example)
//!    drops the file before exec-ing Veld so we get an exact identity
//!    without environment-variable pollution.
//! 2. **`VELD_AGENT_ID` environment variable.** Same whitelist.
//! 3. **Parent-process inspection** via [`sysinfo`]. Best-effort mapping from
//!    common launcher binary names to chat brands. VS Code's `code` binary
//!    can host either Claude or Copilot; without a way to introspect which
//!    extension is active from a child process we default to `"Copilot"` —
//!    naming the more likely tenant rather than silently misattributing to
//!    Claude.
//! 4. **`None`.** Detection failed; the engram still ships, just without a
//!    brand.
//!
//! # The other fields
//!
//! - `worktree_path` — `std::env::current_dir()` normalized to forward
//!   slashes (POSIX form). Same canonical shape on Windows and Unix, so
//!   memories exported across platforms group correctly by worktree.
//! - `branch` — `git -C <worktree_path> rev-parse --abbrev-ref HEAD` with a
//!   1 s wall-clock timeout. `None` on failure / timeout.
//! - `parent_repo` — `git -C <worktree_path> rev-parse --git-common-dir`,
//!   resolved to its parent directory; `None` if it matches the worktree
//!   (i.e. the worktree *is* the main clone) or if git failed.
//! - `vscode_window_id` — `VSCODE_PID`, falling back to
//!   `TERM_PROGRAM_VERSION`; opportunistic.
//! - `started_at` — `Utc::now()` when detection first runs.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use chrono::Utc;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::Deserialize;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};

use crate::memory::facets::AgentSession;

/// Whitelist of accepted chat-brand identifiers.
///
/// Any value produced by the file-or-env path that is *not* on this list is
/// rejected and falls through to parent-process detection. Process-name
/// mapping cannot produce a value outside the whitelist, so the same names
/// (or `None`) come back from every code path. Order is preserved for
/// stability; matching is case-sensitive.
pub const AGENT_ID_WHITELIST: &[&str] = &[
    "Claude", "Copilot", "Cursor", "Aider", "Continue", "unknown",
];

/// Git CLI timeout — branch / parent-repo probes both share this budget.
///
/// One second is generous for a local `rev-parse` and short enough that a
/// hung or non-existent git binary cannot stall the writer path.
const GIT_TIMEOUT: Duration = Duration::from_secs(1);

/// File name prefix for the sentinel agent-id drop file. The full file name
/// is `<prefix><pid>` in `current_dir()`.
const SESSION_FILE_PREFIX: &str = ".veld-agent-session.";

/// Process-wide cache. `None` means "never detected"; `Some(_)` is the
/// cached identity. Wrapped in a [`Mutex`] rather than a bare [`OnceCell`]
/// so [`invalidate_session_cache`] can reset it for tests without unsafe
/// reinitialisation.
static SESSION_CACHE: Lazy<Mutex<Option<AgentSession>>> = Lazy::new(|| Mutex::new(None));

/// Schema of the sentinel file — only `agent_id` is read today. Extra
/// fields are tolerated (`serde(deny_unknown_fields)` deliberately *off*) so
/// future hook drop-files can carry additional context without breaking
/// older Veld binaries.
#[derive(Debug, Clone, Deserialize)]
struct SessionFile {
    #[serde(default)]
    agent_id: Option<String>,
}

/// Detect the current agent session, or return the cached value from this
/// process's first call.
///
/// Always returns an [`AgentSession`] — fields are `None` when the
/// corresponding detection step failed, never silently faked.
pub fn detect_session() -> AgentSession {
    {
        let guard = SESSION_CACHE.lock();
        if let Some(cached) = guard.as_ref() {
            return cached.clone();
        }
    }

    let session = build_session();
    *SESSION_CACHE.lock() = Some(session.clone());
    session
}

/// Drop the cached session so the next [`detect_session`] call re-detects.
///
/// Test-only by intent, but kept `pub` because the test cases live in this
/// file (private) *and* in higher-level integration tests that mutate env
/// vars between cases. Calling this from production code is harmless but
/// pointless — session identity does not change mid-process.
pub fn invalidate_session_cache() {
    *SESSION_CACHE.lock() = None;
}

/// Run the full detection pipeline. Pure function over the environment and
/// filesystem; [`detect_session`] handles caching.
fn build_session() -> AgentSession {
    let worktree_path = current_dir_posix();
    let worktree_str = worktree_path
        .as_ref()
        .and_then(|p| p.to_str().map(|s| s.to_owned()));

    let agent_id = detect_agent_id(worktree_str.as_deref());

    let (branch, parent_repo) = match worktree_str.as_deref() {
        Some(dir) => (git_branch(dir), git_parent_repo(dir, worktree_path.as_deref())),
        None => (None, None),
    };

    let vscode_window_id = env::var("VSCODE_PID")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| env::var("TERM_PROGRAM_VERSION").ok().filter(|s| !s.is_empty()));

    AgentSession {
        worktree_path,
        branch,
        agent_id,
        vscode_window_id,
        started_at: Some(Utc::now()),
        parent_repo,
    }
}

/// Resolve `current_dir()` and normalize to POSIX (forward-slash) form. On
/// Unix the input already uses forward slashes; on Windows we rewrite the
/// backslashes so the serialized facet round-trips identically across
/// platforms.
fn current_dir_posix() -> Option<PathBuf> {
    let cwd = env::current_dir().ok()?;
    Some(to_posix(&cwd))
}

/// Convert a path to forward-slash form. Lossy where the path is not valid
/// UTF-8 (which is exceedingly rare for git worktree roots); in that case we
/// pass the path through unchanged rather than corrupting it.
fn to_posix(path: &Path) -> PathBuf {
    match path.to_str() {
        Some(s) => PathBuf::from(s.replace('\\', "/")),
        None => path.to_path_buf(),
    }
}

// =============================================================================
// agent_id resolution — file -> env -> parent-process -> None
// =============================================================================

fn detect_agent_id(worktree_dir: Option<&str>) -> Option<String> {
    if let Some(dir) = worktree_dir {
        if let Some(id) = read_session_file(dir) {
            return Some(id);
        }
    }
    if let Some(id) = read_env_agent_id() {
        return Some(id);
    }
    parent_process_brand()
}

/// Read `<cwd>/.veld-agent-session.<pid>` and pull `agent_id` from it if the
/// brand is on the whitelist.
fn read_session_file(worktree_dir: &str) -> Option<String> {
    let pid = std::process::id();
    let path = PathBuf::from(worktree_dir).join(format!("{SESSION_FILE_PREFIX}{pid}"));
    let raw = std::fs::read_to_string(&path).ok()?;
    let file: SessionFile = serde_json::from_str(&raw).ok()?;
    let candidate = file.agent_id?;
    whitelist(&candidate)
}

/// Read `VELD_AGENT_ID`, returning the value only if it's on the whitelist.
fn read_env_agent_id() -> Option<String> {
    let raw = env::var("VELD_AGENT_ID").ok()?;
    whitelist(&raw)
}

/// Apply [`AGENT_ID_WHITELIST`] — case-sensitive match, returning the
/// canonical (whitelist-side) spelling so casing drift in callers does not
/// produce drift in stored facets.
fn whitelist(candidate: &str) -> Option<String> {
    AGENT_ID_WHITELIST
        .iter()
        .find(|allowed| **allowed == candidate)
        .map(|s| (*s).to_owned())
}

/// Best-effort parent-process brand inference via [`sysinfo`]. Returns
/// `None` if the parent can't be identified or doesn't match a known
/// launcher.
///
/// Mapping:
/// - `code` / `code.exe`     → `"Copilot"` (we cannot tell from a child
///   process whether the Claude or Copilot VS Code extension is the active
///   driver, so we name the more common tenant rather than misattribute to
///   Claude).
/// - `cursor` / `cursor.exe` → `"Cursor"`.
/// - `claude` / `claude-code` / `claude-cli` (with optional `.exe`) →
///   `"Claude"`.
fn parent_process_brand() -> Option<String> {
    let mut system = System::new();
    let self_pid = Pid::from_u32(std::process::id());
    // Minimal refresh — just enough to read the executable name. `exe` is
    // populated lazily, so `OnlyIfNotSet` is the cheap choice on Windows
    // where reading the exe path requires opening the process handle.
    let refresh = ProcessRefreshKind::new()
        .with_cmd(UpdateKind::OnlyIfNotSet)
        .with_exe(UpdateKind::OnlyIfNotSet);
    system.refresh_processes_specifics(ProcessesToUpdate::Some(&[self_pid]), false, refresh);

    let parent_pid = system.process(self_pid)?.parent()?;
    system.refresh_processes_specifics(ProcessesToUpdate::Some(&[parent_pid]), false, refresh);
    let parent = system.process(parent_pid)?;

    let name = parent.name().to_string_lossy().to_lowercase();
    let stem = strip_exe(&name);
    match stem {
        "code" => Some("Copilot".to_owned()),
        "cursor" => Some("Cursor".to_owned()),
        "claude" | "claude-code" | "claude-cli" => Some("Claude".to_owned()),
        _ => None,
    }
}

/// Strip a trailing `.exe` (Windows binary suffix). Caller has already
/// lowercased.
fn strip_exe(name: &str) -> &str {
    name.strip_suffix(".exe").unwrap_or(name)
}

// =============================================================================
// git probes — bounded shell-out to local git
// =============================================================================

/// `git -C <dir> rev-parse --abbrev-ref HEAD`, with a 1 s timeout.
fn git_branch(dir: &str) -> Option<String> {
    let out = run_git_with_timeout(&["-C", dir, "rev-parse", "--abbrev-ref", "HEAD"])?;
    let trimmed = out.trim();
    if trimmed.is_empty() || trimmed == "HEAD" {
        // Detached HEAD or empty output — branch is genuinely unknown.
        return None;
    }
    Some(trimmed.to_owned())
}

/// `git -C <dir> rev-parse --git-common-dir`, resolved to its parent
/// directory. Returns `None` when the result *is* the worktree (the
/// worktree is the main clone) or when git failed.
fn git_parent_repo(dir: &str, worktree: Option<&Path>) -> Option<PathBuf> {
    let out = run_git_with_timeout(&["-C", dir, "rev-parse", "--git-common-dir"])?;
    let common = out.trim();
    if common.is_empty() {
        return None;
    }

    // `--git-common-dir` can be relative ("./.git") or absolute. Resolve
    // against the worktree dir so we always end up with an absolute path.
    let common_path = {
        let p = PathBuf::from(common);
        if p.is_absolute() {
            p
        } else {
            PathBuf::from(dir).join(p)
        }
    };
    let main_worktree = common_path.parent()?.to_path_buf();
    let main_posix = to_posix(&main_worktree);

    // Suppress when this *is* the main worktree.
    if let Some(wt) = worktree {
        if to_posix(wt) == main_posix {
            return None;
        }
    }
    Some(main_posix)
}

/// Run `git <args>` and return stdout as UTF-8, killing the child if it
/// takes longer than [`GIT_TIMEOUT`]. Returns `None` on spawn failure,
/// timeout, non-zero exit, or non-UTF-8 output.
fn run_git_with_timeout(args: &[&str]) -> Option<String> {
    // Build the argv on the calling thread (cheap, avoids `Send`-ing &str refs
    // into the spawned closure).
    let owned: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let result = Command::new("git").args(&owned).output();
        // Receiver may already be dropped if the parent timed out; ignore the
        // send error in that case.
        let _ = tx.send(result);
    });

    let result = rx.recv_timeout(GIT_TIMEOUT).ok()?.ok()?;
    if !result.status.success() {
        return None;
    }
    String::from_utf8(result.stdout).ok()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::fs;
    use std::sync::Mutex as StdMutex;

    /// All env-mutating tests serialize on this lock. Rust's test harness
    /// runs tests in parallel; `set_var` is process-global, so without a
    /// guard one test would clobber another's environment.
    static ENV_GUARD: StdMutex<()> = StdMutex::new(());

    /// Per-test fixture: takes the env guard, snapshots and clears the
    /// VELD_AGENT_ID env var, switches `cwd` to a fresh temp dir, and
    /// restores everything (env + cwd) on drop.
    struct Fixture {
        _lock: std::sync::MutexGuard<'static, ()>,
        original_cwd: PathBuf,
        original_agent_id: Option<OsString>,
        _tempdir: tempfile::TempDir,
    }

    impl Fixture {
        fn new() -> Self {
            // Poisoned guard is acceptable — the previous test panicked, but
            // the env is whatever that test left, and we're about to reset
            // it. Either side of the result is a valid guard.
            let lock = ENV_GUARD.lock().unwrap_or_else(|poison| poison.into_inner());
            let original_cwd = env::current_dir().expect("cwd readable");
            let original_agent_id = env::var_os("VELD_AGENT_ID");
            // Tests run on a single thread inside ENV_GUARD; no other code in
            // this process is touching VELD_AGENT_ID concurrently.
            env::remove_var("VELD_AGENT_ID");

            let tempdir = tempfile::tempdir().expect("create tempdir");
            env::set_current_dir(tempdir.path()).expect("chdir to tempdir");
            invalidate_session_cache();

            Self {
                _lock: lock,
                original_cwd,
                original_agent_id,
                _tempdir: tempdir,
            }
        }

        fn pid_file_path(&self) -> PathBuf {
            self._tempdir
                .path()
                .join(format!("{SESSION_FILE_PREFIX}{}", std::process::id()))
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            // Restore env var. Still holding ENV_GUARD; single-threaded with
            // respect to other tests in this module.
            if let Some(val) = self.original_agent_id.take() {
                env::set_var("VELD_AGENT_ID", val);
            } else {
                env::remove_var("VELD_AGENT_ID");
            }
            let _ = env::set_current_dir(&self.original_cwd);
            invalidate_session_cache();
        }
    }

    #[test]
    fn session_file_takes_precedence_over_env_and_parent() {
        let fx = Fixture::new();
        // Env says Cursor; file says Claude — file wins.
        env::set_var("VELD_AGENT_ID", "Cursor");
        fs::write(fx.pid_file_path(), r#"{"agent_id":"Claude"}"#).unwrap();
        invalidate_session_cache();

        let session = detect_session();
        assert_eq!(session.agent_id.as_deref(), Some("Claude"));
    }

    #[test]
    fn env_var_fills_in_when_file_absent() {
        let _fx = Fixture::new();
        env::set_var("VELD_AGENT_ID", "Cursor");
        invalidate_session_cache();

        let session = detect_session();
        assert_eq!(session.agent_id.as_deref(), Some("Cursor"));
    }

    #[test]
    fn whitelist_rejects_unknown_brand() {
        let _fx = Fixture::new();
        // "evil" is not on the whitelist — env path returns None, falling
        // through to parent-process detection. The parent of `cargo test`
        // is `cargo` (or a shell), neither of which maps to a known brand,
        // so the final agent_id is None.
        env::set_var("VELD_AGENT_ID", "evil");
        invalidate_session_cache();

        let session = detect_session();
        assert!(
            session.agent_id.is_none(),
            "expected None for non-whitelisted brand, got {:?}",
            session.agent_id
        );
    }

    #[test]
    fn worktree_path_uses_forward_slashes() {
        let _fx = Fixture::new();
        invalidate_session_cache();

        let session = detect_session();
        let path = session
            .worktree_path
            .as_ref()
            .expect("worktree_path detected");
        let s = path.to_str().expect("UTF-8 worktree path");
        assert!(
            !s.contains('\\'),
            "worktree_path contained a backslash: {s:?}"
        );
    }

    #[test]
    fn invalidate_cache_re_runs_detection() {
        let _fx = Fixture::new();
        env::set_var("VELD_AGENT_ID", "Claude");
        invalidate_session_cache();
        let first = detect_session();
        assert_eq!(first.agent_id.as_deref(), Some("Claude"));

        // Second call without invalidation still returns the cached value
        // even though the env changed.
        env::set_var("VELD_AGENT_ID", "Cursor");
        let cached = detect_session();
        assert_eq!(cached.agent_id.as_deref(), Some("Claude"));

        // After invalidation we see the new env.
        invalidate_session_cache();
        let fresh = detect_session();
        assert_eq!(fresh.agent_id.as_deref(), Some("Cursor"));
    }

    #[test]
    fn whitelist_round_trip_canonical_spelling() {
        assert_eq!(whitelist("Claude").as_deref(), Some("Claude"));
        // Case-sensitive: lower-case is rejected.
        assert!(whitelist("claude").is_none());
        assert!(whitelist("").is_none());
        assert!(whitelist("Anthropic").is_none());
    }

    #[test]
    fn strip_exe_is_idempotent_off_windows() {
        assert_eq!(strip_exe("code"), "code");
        assert_eq!(strip_exe("code.exe"), "code");
    }

    #[test]
    fn to_posix_replaces_backslashes() {
        let p = Path::new(r"C:\foo\bar");
        let posix = to_posix(p);
        assert_eq!(posix.to_str(), Some("C:/foo/bar"));
    }
}
