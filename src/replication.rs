//! Live replication between veld instances.
//!
//! A follower tails a leader's per-user intent logs over HTTP and applies every
//! frame to its own canonical store, then journals the same frame locally so its
//! projections (vector, BM25, SQLite, wiki) and its own log follow. The intent log
//! already holds a full memory snapshot per `Remember`/`Update`, so a follower that
//! starts empty converges on the leader's memories by replaying from LSN 0, ids and
//! timestamps intact. Embeddings travel inside the snapshot: the follower never
//! re-embeds.
//!
//! Scope and limits, stated rather than implied:
//! - One direction. A follower is read-mostly; writes made on it are not sent
//!   back, and two instances following each other would echo frames forever, so
//!   `follow` refuses a source that reports following us.
//! - Memories only. Todos, reminders, projects, context blocks and dials are not
//!   journaled, so they do not replicate. The knowledge graph is rebuilt from the
//!   memories by the follower's own consolidation.
//! - Frames leave the leader decrypted (its journal cryptor opens them) and are
//!   sealed again by the follower's own keystore on append. The admin key and API
//!   key ride every request, so a leader that is not on this host must be https.
//!
//! Each tenant is synced on its own: a tenant whose page fails is retried with its
//! own backoff while the others carry on, and a tenant whose leader log moved
//! backwards (restore, point-in-time rollback, recreated log) is marked diverged and
//! left alone rather than silently skipping what it missed.
//!
//! Change detection is by the byte length of each user's `intent.log`, read from
//! the filesystem without opening the user's store, so an idle poll costs one
//! `stat` per user on the leader.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::handlers::state::MultiUserMemoryManager;
use crate::intent_log::{payload, IntentPayload, LogMode};
use crate::memory::types::MemoryId;
use crate::memory::ForgetCriteria;

pub const ENV_SOURCE: &str = "VELD_REPLICATE_FROM";
pub const ENV_ADMIN_KEY: &str = "VELD_REPLICATE_ADMIN_KEY";
/// The leader's per-request API key; defaults to this server's own first key when
/// the leader is on this host, which is right whenever both share one `config.toml`.
pub const ENV_API_KEY: &str = "VELD_REPLICATE_API_KEY";
pub const ENV_INTERVAL_SECS: &str = "VELD_REPLICATE_INTERVAL_SECS";
/// Set to `1` to follow a leader on another host over plain http. The admin key and
/// API key then cross the network in cleartext; meant for a private link only.
pub const ENV_INSECURE_HTTP: &str = "VELD_REPLICATE_INSECURE_HTTP";
pub const DEFAULT_INTERVAL: Duration = Duration::from_secs(2);
/// Frames per page of the feed; a page is one HTTP round trip.
pub const FEED_LIMIT: usize = 500;
/// Largest leader reply the follower will hold. A 500-frame page of snapshots with
/// embeddings is about 8 MB; anything near this is not a page.
const MAX_REPLY_BYTES: usize = 256 * 1024 * 1024;
/// How much of a leader's error body is carried into an error message.
const MAX_ECHO_CHARS: usize = 300;
const MAX_BACKOFF: Duration = Duration::from_secs(300);
const CURSOR_DIR: &str = "replication";

/// One user's intent log on the leader, as the follower sees it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserLogInfo {
    pub user_id: String,
    pub log_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedRecord {
    pub lsn: u64,
    /// The frame's payload bytes (decrypted, bincode `IntentPayload`), base64.
    pub frame_b64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedPage {
    pub user_id: String,
    pub records: Vec<FeedRecord>,
    /// The LSN the leader will assign next; `records` end before it.
    pub head_lsn: u64,
    pub more: bool,
    /// Identity of the leader's log file: the creation time in its header, 0 for a
    /// legacy log without one. A different value means the log was recreated.
    #[serde(default)]
    pub log_created: u64,
}

/// What a follower follows. The keys never leave the process.
#[derive(Debug, Clone, Serialize)]
pub struct FollowTarget {
    pub source: String,
    #[serde(skip)]
    pub admin_key: String,
    #[serde(skip)]
    pub api_key: String,
    pub since: DateTime<Utc>,
}

/// This server's own first API key, from the same variables the auth
/// middleware reads.
pub fn own_api_key() -> Option<String> {
    ["VELD_API_KEYS", "VELD_API_KEY", "VELD_DEV_API_KEY"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .flat_map(|v| v.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>())
        .find(|k| !k.is_empty())
}

/// The API key to present to `source` when the caller gave none: this server's
/// own, but only for a leader on this host. A remote leader must be given its key
/// explicitly, so a mistyped or hostile URL never receives ours.
pub fn default_api_key_for(source: &str) -> Option<String> {
    if is_loopback_source(source) {
        own_api_key()
    } else {
        None
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserProgress {
    pub applied_lsn: Option<u64>,
    /// Leader frames consumed.
    pub applied_records: u64,
    /// Frames written to this follower's own log after coalescing.
    #[serde(default)]
    pub journaled_records: u64,
    /// Leader log length at the last poll that found nothing new.
    pub source_bytes: u64,
    /// The leader's next LSN at the last page; lag is `leader_head_lsn - 1 - applied_lsn`.
    #[serde(default)]
    pub leader_head_lsn: Option<u64>,
    /// The leader log identity this cursor belongs to (see [`FeedPage::log_created`]).
    #[serde(default)]
    pub log_created: Option<u64>,
    /// Set when the leader's log no longer contains what this follower applied.
    /// The tenant is not synced again until its local data is removed.
    #[serde(default)]
    pub diverged: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub failures: u32,
    #[serde(skip)]
    pub retry_at: Option<Instant>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FollowStatus {
    pub target: Option<FollowTarget>,
    /// Bumped by every follow and unfollow; a poll started under an older value
    /// stops at its next page and writes nothing.
    pub generation: u64,
    pub polls: u64,
    pub pages: u64,
    pub last_poll: Option<DateTime<Utc>>,
    pub last_applied: Option<DateTime<Utc>>,
    /// The last failure to reach the leader at all; per-tenant failures are on the tenant.
    pub last_error: Option<String>,
    pub users: BTreeMap<String, UserProgress>,
}

impl FollowStatus {
    /// Record one applied page for `user_id` if `generation` is still current.
    /// Returns the cursor to persist, or None when the poll is stale.
    fn record_page(
        &mut self,
        generation: u64,
        user_id: &str,
        page: &FeedPage,
        last_lsn: u64,
        received: u64,
        journaled: u64,
    ) -> Option<BTreeMap<String, UserProgress>> {
        if self.generation != generation {
            return None;
        }
        let entry = self.users.entry(user_id.to_string()).or_default();
        entry.applied_lsn = Some(last_lsn);
        entry.applied_records += received;
        entry.journaled_records += journaled;
        entry.leader_head_lsn = Some(page.head_lsn);
        entry.log_created = Some(page.log_created);
        self.pages += 1;
        self.last_applied = Some(Utc::now());
        Some(self.users.clone())
    }
}

static FOLLOW: LazyLock<RwLock<FollowStatus>> =
    LazyLock::new(|| RwLock::new(FollowStatus::default()));

pub fn status() -> FollowStatus {
    FOLLOW.read().clone()
}

fn is_current(generation: u64) -> bool {
    FOLLOW.read().generation == generation
}

/// Host part of an http(s) URL, without brackets for IPv6.
fn source_host(source: &str) -> &str {
    let rest = source
        .strip_prefix("https://")
        .or_else(|| source.strip_prefix("http://"))
        .unwrap_or(source);
    let authority = rest.split('/').next().unwrap_or("");
    if let Some(v6) = authority.strip_prefix('[') {
        return v6.split(']').next().unwrap_or("");
    }
    authority.rsplit_once(':').map_or(authority, |(h, _)| h)
}

fn source_port(source: &str) -> Option<u16> {
    let rest = source
        .strip_prefix("https://")
        .or_else(|| source.strip_prefix("http://"))?;
    let authority = rest.split('/').next().unwrap_or("");
    let after_host = match authority.strip_prefix('[') {
        Some(v6) => v6.split_once(']').map(|(_, tail)| tail).unwrap_or(""),
        None => authority.rsplit_once(':').map_or("", |(_, p)| p),
    };
    match after_host.trim_start_matches(':') {
        "" if source.starts_with("https://") => Some(443),
        "" => Some(80),
        p => p.parse().ok(),
    }
}

pub fn is_loopback_source(source: &str) -> bool {
    let host = source_host(source);
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

/// Whether `source` addresses this instance: the same URL, or any loopback name
/// with this instance's port while this instance listens on loopback or all
/// interfaces.
pub fn is_same_instance(source: &str, own_host: &str, own_port: u16) -> bool {
    let own_reachable_locally = matches!(own_host, "0.0.0.0" | "::" | "[::]")
        || is_loopback_source(&format!("http://{own_host}"));
    own_reachable_locally && is_loopback_source(source) && source_port(source) == Some(own_port)
}

fn normalize_source_with(raw: &str, allow_insecure_http: bool) -> Result<String> {
    let s = raw.trim().trim_end_matches('/');
    let secure = s.starts_with("https://");
    if !(secure || s.starts_with("http://")) {
        bail!("source must be an http(s) URL, got '{raw}'");
    }
    let rest = s.split_once("://").map_or("", |(_, r)| r);
    if rest.split('/').next().unwrap_or("").contains('@') {
        bail!("source must not carry credentials in the URL");
    }
    if source_host(s).is_empty() {
        bail!("source has no host: '{raw}'");
    }
    if !secure && !is_loopback_source(s) && !allow_insecure_http {
        bail!(
            "a leader on another host must be https: the admin key and API key ride every \
             request. Set {ENV_INSECURE_HTTP}=1 to accept plain http on a private link"
        );
    }
    Ok(s.to_string())
}

pub fn normalize_source(raw: &str) -> Result<String> {
    let insecure = std::env::var(ENV_INSECURE_HTTP).is_ok_and(|v| v.trim() == "1");
    normalize_source_with(raw, insecure)
}

/// Leader side: every user's log length, by `stat` alone.
pub fn user_logs(manager: &MultiUserMemoryManager) -> Vec<UserLogInfo> {
    let mut out: Vec<UserLogInfo> = manager
        .list_users()
        .into_iter()
        .filter_map(|user_id| {
            let dir = manager.user_dir(&user_id).ok()?;
            let log_bytes = std::fs::metadata(dir.join("intent.log"))
                .map(|m| m.len())
                .unwrap_or(0);
            Some(UserLogInfo { user_id, log_bytes })
        })
        .collect();
    out.sort_by(|a, b| a.user_id.cmp(&b.user_id));
    out
}

/// Whether `user_id` already has an intent log on this instance. The feed only
/// serves existing logs: opening a writer for an unknown tenant would create one.
pub fn has_log(manager: &MultiUserMemoryManager, user_id: &str) -> bool {
    manager
        .user_dir(user_id)
        .map(|d| d.join("intent.log").is_file())
        .unwrap_or(false)
}

/// Leader side: frames of `user_id` after `after_lsn`, at most `limit`. Blocking.
pub fn feed(
    manager: &MultiUserMemoryManager,
    user_id: &str,
    after_lsn: Option<u64>,
    limit: usize,
) -> Result<FeedPage> {
    let writer = manager.get_user_journaled_writer(user_id)?;
    let (iter, head_lsn, log_created) = {
        let guard = writer.lock();
        let log_created = match guard.log().mode() {
            LogMode::Versioned {
                created_unix_secs, ..
            } => created_unix_secs,
            LogMode::Legacy => 0,
        };
        (guard.log().iter()?, guard.log().next_lsn().0, log_created)
    };
    let floor = after_lsn.map(|l| l + 1).unwrap_or(0);
    let mut records = Vec::new();
    let mut more = false;
    for frame in iter {
        let record = frame.context("reading the intent log")?;
        if record.lsn.0 < floor {
            continue;
        }
        if records.len() >= limit {
            more = true;
            break;
        }
        records.push(FeedRecord {
            lsn: record.lsn.0,
            frame_b64: base64::engine::general_purpose::STANDARD.encode(&record.payload),
        });
    }
    Ok(FeedPage {
        user_id: user_id.to_string(),
        records,
        head_lsn,
        more,
        log_created,
    })
}

/// Why a tenant's cursor no longer fits the leader's log, or None when it does.
/// The leader's head is the next LSN it will assign, so a follower that applied
/// LSN `a` needs a head above `a`; a changed log identity means the file was
/// recreated and its LSNs name different frames.
pub fn divergence(progress: &UserProgress, page: &FeedPage) -> Option<String> {
    if let (Some(seen), true) = (progress.log_created, page.log_created != 0) {
        if seen != 0 && seen != page.log_created {
            return Some(format!(
                "the leader's log for this tenant was recreated (created {} -> {})",
                seen, page.log_created
            ));
        }
    }
    let applied = progress.applied_lsn?;
    if page.head_lsn <= applied {
        return Some(format!(
            "the leader's log ends before what this follower applied (leader next lsn {}, applied {}): \
             restored, rolled back or truncated",
            page.head_lsn, applied
        ));
    }
    None
}

/// A page of leader frames reduced to what changes the follower's state: for each
/// memory id, the last `Remember`/`Update` snapshot in the page and every frame
/// after it. Earlier snapshots of the same id are superseded, so applying only the
/// last one is the same final state at a fraction of the writes; logs are
/// Update-heavy because every access and consolidation re-journals a memory.
pub fn coalesce(frames: Vec<(u64, IntentPayload)>) -> Vec<(u64, IntentPayload)> {
    let mut order: Vec<String> = Vec::new();
    let mut by_id: HashMap<String, Vec<(u64, IntentPayload)>> = HashMap::new();
    for (lsn, payload) in frames {
        let id = payload.memory_id().to_string();
        if !by_id.contains_key(&id) {
            order.push(id.clone());
        }
        by_id.entry(id).or_default().push((lsn, payload));
    }
    let mut out = Vec::new();
    for id in order {
        let mut frames = by_id.remove(&id).unwrap_or_default();
        let last_snapshot = frames.iter().rposition(|(_, p)| {
            matches!(p, IntentPayload::Remember { .. } | IntentPayload::Update { .. })
        });
        if let Some(i) = last_snapshot {
            frames.drain(..i);
        }
        out.extend(frames);
    }
    out
}

/// Refuse a page whose frames name a different tenant than the page they came in.
fn check_tenant(user_id: &str, frames: &[(u64, IntentPayload)]) -> Result<()> {
    if let Some((lsn, p)) = frames.iter().find(|(_, p)| p.user_id() != user_id) {
        bail!(
            "frame lsn {lsn} in the page for tenant '{user_id}' names tenant '{}'",
            p.user_id()
        );
    }
    Ok(())
}

/// Follower side: apply a page of leader frames to the canonical store, then
/// journal each applied frame here so the local projections and log follow. The
/// page is coalesced first; the BM25 projection indexes snapshots as their frames
/// are journaled, on its own commit cadence. Idempotent:
/// a snapshot is an upsert, a forget of an unknown id is a no-op. Returns the
/// number of frames journaled. Blocking.
pub fn apply_replicated_page(
    manager: &MultiUserMemoryManager,
    user_id: &str,
    frames: Vec<(u64, IntentPayload)>,
) -> Result<u64> {
    check_tenant(user_id, &frames)?;
    let frames = coalesce(frames);
    if frames.is_empty() {
        return Ok(0);
    }
    let memory_sys = manager.get_user_memory(user_id)?;
    let mut snapshots: Vec<crate::memory::types::Memory> = Vec::new();
    for (lsn, payload) in &frames {
        if let IntentPayload::Remember {
            memory_bincode,
            memory_id,
            ..
        }
        | IntentPayload::Update {
            memory_bincode,
            memory_id,
            ..
        } = payload
        {
            // The snapshot ages like the store's own records: frames journaled before a
            // layout change need the legacy decode arms, the same ones the projections use.
            let memory = crate::memory::storage::decode_memory_snapshot(memory_bincode)
                .map_err(|e| anyhow!("decoding the memory snapshot at lsn {lsn}: {e}"))?;
            if memory.id.0.to_string() != *memory_id {
                bail!(
                    "frame lsn {lsn} names memory {memory_id} but its snapshot is memory {}",
                    memory.id.0
                );
            }
            snapshots.push(memory);
        }
    }
    memory_sys.read().upsert_memories(&snapshots)?;
    let mut journaled = 0u64;
    for (_, payload) in &frames {
        apply_non_snapshot(manager, user_id, payload)?;
        manager.journal_and_apply(user_id, payload)?;
        journaled += 1;
    }
    Ok(journaled)
}

/// The store-side effect of a `Forget` or `Anchor` frame; snapshots are handled in
/// bulk by [`apply_replicated_page`].
fn apply_non_snapshot(
    manager: &MultiUserMemoryManager,
    user_id: &str,
    payload: &IntentPayload,
) -> Result<()> {
    let memory_sys = manager.get_user_memory(user_id)?;
    match payload {
        IntentPayload::Remember { .. } | IntentPayload::Update { .. } => {}
        IntentPayload::Forget { memory_id, .. } => {
            let id = MemoryId(uuid::Uuid::parse_str(memory_id)?);
            let guard = memory_sys.read();
            if guard.get_memory(&id).is_ok() {
                guard.forget(ForgetCriteria::ById(id))?;
            }
        }
        IntentPayload::Anchor {
            memory_id,
            importance,
            ..
        } => {
            let id = MemoryId(uuid::Uuid::parse_str(memory_id)?);
            let guard = memory_sys.read();
            if let Ok(memory) = guard.get_memory(&id) {
                // The frame carries the importance after the change, not the flag:
                // a rise is an anchor, a fall is its release.
                if *importance > memory.importance() {
                    memory.set_anchored(true);
                } else if *importance < memory.importance() {
                    memory.set_anchored(false);
                }
                memory.set_importance(*importance);
                guard.update_memory(&memory)?;
            }
        }
    }
    Ok(())
}

fn cursor_path(base: &Path, source: &str) -> PathBuf {
    let slug: String = source
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    base.join(CURSOR_DIR).join(format!("{slug}.json"))
}

/// The saved cursor for a source. A missing file is a fresh follow; a file that
/// cannot be read or parsed is set aside and reported, then treated as fresh, so
/// the tenants replay from the start instead of the follower stalling.
fn load_cursor(path: &Path) -> BTreeMap<String, UserProgress> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return BTreeMap::new(),
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "replication: cursor unreadable; every tenant replays from the start");
            return BTreeMap::new();
        }
    };
    match serde_json::from_str(&text) {
        Ok(users) => users,
        Err(e) => {
            let aside = path.with_extension("json.unreadable");
            let _ = std::fs::rename(path, &aside);
            tracing::warn!(path = %path.display(), aside = %aside.display(), error = %e, "replication: cursor does not parse; set aside, every tenant replays from the start");
            BTreeMap::new()
        }
    }
}

fn save_cursor(path: &Path, users: &BTreeMap<String, UserProgress>) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(users)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Start (or switch) following `source`. The cursor saved for that source is
/// resumed, so a follower that was stopped picks up where it left off. A poll still
/// running for the previous target stops at its next page.
pub fn follow(
    base: &Path,
    source: &str,
    admin_key: &str,
    api_key: Option<&str>,
) -> Result<FollowStatus> {
    let source = normalize_source(source)?;
    let api_key = api_key
        .map(str::to_string)
        .filter(|k| !k.trim().is_empty())
        .or_else(|| default_api_key_for(&source))
        .ok_or_else(|| {
            anyhow!("no API key for the leader {source}: pass its key, or set {ENV_API_KEY}")
        })?;
    let users = load_cursor(&cursor_path(base, &source));
    let mut status = FOLLOW.write();
    let generation = status.generation + 1;
    *status = FollowStatus {
        target: Some(FollowTarget {
            source,
            admin_key: admin_key.to_string(),
            api_key,
            since: Utc::now(),
        }),
        generation,
        users,
        ..FollowStatus::default()
    };
    Ok(status.clone())
}

/// Stop following. A poll in progress stops at its next page; the cursor is kept.
pub fn unfollow() -> FollowStatus {
    let mut status = FOLLOW.write();
    status.target = None;
    status.generation += 1;
    status.clone()
}

/// `VELD_REPLICATE_FROM` + `VELD_REPLICATE_ADMIN_KEY` at boot.
pub fn follow_from_env(base: &Path) -> Result<Option<FollowStatus>> {
    let Some(source) = std::env::var(ENV_SOURCE)
        .ok()
        .filter(|s| !s.trim().is_empty())
    else {
        return Ok(None);
    };
    let admin_key = std::env::var(ENV_ADMIN_KEY)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| anyhow!("{ENV_SOURCE} is set but {ENV_ADMIN_KEY} is not"))?;
    let api_key = std::env::var(ENV_API_KEY).ok();
    follow(base, &source, &admin_key, api_key.as_deref()).map(Some)
}

pub fn interval_from_env() -> Duration {
    std::env::var(ENV_INTERVAL_SECS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|s| *s > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_INTERVAL)
}

/// Wait before retry number `failures`: the poll interval doubled per failure,
/// capped at five minutes.
pub fn backoff(interval: Duration, failures: u32) -> Duration {
    let factor = 1u32 << failures.min(16);
    interval.saturating_mul(factor).min(MAX_BACKOFF)
}

fn percent_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// At most [`MAX_ECHO_CHARS`] characters of a leader's error body, on a char
/// boundary.
fn echo(body: &str) -> String {
    let trimmed = body.trim();
    let mut out: String = trimmed.chars().take(MAX_ECHO_CHARS).collect();
    if out.len() < trimmed.len() {
        out.push('…');
    }
    out
}

async fn read_capped(mut resp: reqwest::Response, cap: usize) -> Result<Vec<u8>> {
    if let Some(len) = resp.content_length() {
        if len > cap as u64 {
            bail!("the leader's reply is {len} bytes, over the {cap}-byte limit");
        }
    }
    let mut buf = Vec::new();
    while let Some(chunk) = resp.chunk().await? {
        if buf.len() + chunk.len() > cap {
            bail!("the leader's reply passed the {cap}-byte limit");
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

async fn get_json<T: serde::de::DeserializeOwned>(
    http: &reqwest::Client,
    target: &FollowTarget,
    path: &str,
) -> Result<T> {
    let url = format!("{}{}", target.source, path);
    let resp = http
        .get(&url)
        .header("X-API-Key", &target.api_key)
        .header(crate::handlers::admin::ADMIN_API_KEY_HEADER, &target.admin_key)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    let body = read_capped(resp, MAX_REPLY_BYTES)
        .await
        .with_context(|| format!("GET {url}"))?;
    if !status.is_success() {
        bail!("GET {url}: {status}: {}", echo(&String::from_utf8_lossy(&body)));
    }
    serde_json::from_slice(&body).with_context(|| format!("GET {url}: decoding the reply"))
}

/// The leader's own view: its users' logs and whom it follows, from its status route.
#[derive(Debug, Deserialize)]
struct LeaderStatus {
    users: Vec<UserLogInfo>,
    #[serde(default)]
    following: Option<serde_json::Value>,
}

/// What the leader reports following, if anything: used to refuse a loop.
pub async fn source_follows(
    source: &str,
    admin_key: &str,
    api_key: &str,
) -> Result<Option<String>> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    let target = FollowTarget {
        source: normalize_source(source)?,
        admin_key: admin_key.to_string(),
        api_key: api_key.to_string(),
        since: Utc::now(),
    };
    let leader: LeaderStatus = get_json(&http, &target, "/api/admin/replication/status").await?;
    Ok(leader
        .following
        .and_then(|f| f.get("target")?.get("source")?.as_str().map(str::to_string)))
}

/// Mark a tenant diverged and persist the cursor, if the poll is still current.
fn mark_diverged(generation: u64, cursor: &Path, user_id: &str, reason: &str) -> Result<()> {
    let users = {
        let mut status = FOLLOW.write();
        if status.generation != generation {
            return Ok(());
        }
        let entry = status.users.entry(user_id.to_string()).or_default();
        if entry.diverged.is_none() {
            tracing::error!(
                tenant = %user_id,
                reason,
                "replication: tenant diverged from the leader and is no longer synced; \
                 stop this follower, remove the tenant's directory here, and it replays from the start"
            );
        }
        entry.diverged = Some(reason.to_string());
        status.users.clone()
    };
    save_cursor(cursor, &users)
}

/// Sync one tenant from its cursor to the leader's head, page by page, persisting
/// the cursor after each page. Returns the frames consumed.
async fn sync_user(
    manager: &Arc<MultiUserMemoryManager>,
    http: &reqwest::Client,
    target: &FollowTarget,
    generation: u64,
    cursor: &Path,
    info: &UserLogInfo,
) -> Result<u64> {
    let mut consumed = 0u64;
    loop {
        if !is_current(generation) {
            return Ok(consumed);
        }
        let progress = FOLLOW
            .read()
            .users
            .get(&info.user_id)
            .cloned()
            .unwrap_or_default();
        let path = format!(
            "/api/admin/replication/records?user_id={}&after_lsn={}&limit={FEED_LIMIT}",
            percent_encode(&info.user_id),
            progress.applied_lsn.map(|l| l.to_string()).unwrap_or_default()
        );
        let page: FeedPage = get_json(http, target, &path).await?;
        if let Some(reason) = divergence(&progress, &page) {
            mark_diverged(generation, cursor, &info.user_id, &reason)?;
            return Ok(consumed);
        }
        let Some(last_lsn) = page.records.last().map(|r| r.lsn) else {
            break;
        };
        let mut frames = Vec::with_capacity(page.records.len());
        for record in &page.records {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&record.frame_b64)
                .context("decoding a feed frame")?;
            let payload = payload::decode(&bytes)
                .map_err(|e| anyhow!("decoding frame lsn {}: {e}", record.lsn))?;
            frames.push((record.lsn, payload));
        }
        let received = frames.len() as u64;
        let user_id = info.user_id.clone();
        let m = Arc::clone(manager);
        let journaled =
            tokio::task::spawn_blocking(move || apply_replicated_page(&m, &user_id, frames))
                .await
                .map_err(|e| anyhow!("apply task: {e}"))?
                .with_context(|| format!("applying lsns up to {last_lsn} of {}", info.user_id))?;
        let saved = FOLLOW.write().record_page(
            generation,
            &info.user_id,
            &page,
            last_lsn,
            received,
            journaled,
        );
        let Some(users) = saved else {
            return Ok(consumed);
        };
        save_cursor(cursor, &users)?;
        consumed += received;
        if !page.more {
            break;
        }
    }
    let users = {
        let mut status = FOLLOW.write();
        if status.generation != generation {
            return Ok(consumed);
        }
        let entry = status.users.entry(info.user_id.clone()).or_default();
        entry.source_bytes = info.log_bytes;
        status.users.clone()
    };
    save_cursor(cursor, &users)?;
    Ok(consumed)
}

/// One pass over every tenant the leader lists. A tenant that fails is retried on
/// its own backoff and does not hold up the others. Fails only when the leader's
/// status cannot be read.
async fn poll_once(
    manager: &Arc<MultiUserMemoryManager>,
    http: &reqwest::Client,
    target: &FollowTarget,
    generation: u64,
    interval: Duration,
) -> Result<u64> {
    let leader: LeaderStatus = get_json(http, target, "/api/admin/replication/status").await?;
    let cursor = cursor_path(&manager.base_path, &target.source);
    let mut consumed = 0u64;
    for info in &leader.users {
        if !is_current(generation) {
            break;
        }
        let progress = FOLLOW
            .read()
            .users
            .get(&info.user_id)
            .cloned()
            .unwrap_or_default();
        if progress.diverged.is_some() {
            let local_gone = manager
                .user_dir(&info.user_id)
                .map(|d| !d.exists())
                .unwrap_or(false);
            if !local_gone {
                continue;
            }
            tracing::info!(tenant = %info.user_id, "replication: diverged tenant's local data removed; replaying it from the start");
            let mut status = FOLLOW.write();
            if status.generation != generation {
                break;
            }
            status.users.remove(&info.user_id);
        } else {
            let backing_off = progress.retry_at.is_some_and(|at| Instant::now() < at);
            let quiet = progress.last_error.is_none()
                && progress.applied_lsn.is_some()
                && progress.source_bytes == info.log_bytes;
            if backing_off || quiet {
                continue;
            }
        }
        match sync_user(manager, http, target, generation, &cursor, info).await {
            Ok(n) => {
                consumed += n;
                let mut status = FOLLOW.write();
                if status.generation == generation {
                    if let Some(entry) = status.users.get_mut(&info.user_id) {
                        if entry.last_error.take().is_some() {
                            tracing::info!(tenant = %info.user_id, "replication: tenant recovered");
                        }
                        entry.failures = 0;
                        entry.retry_at = None;
                    }
                }
            }
            Err(e) => {
                let message = format!("{e:#}");
                let mut status = FOLLOW.write();
                if status.generation != generation {
                    break;
                }
                let entry = status.users.entry(info.user_id.clone()).or_default();
                if entry.last_error.as_deref() != Some(message.as_str()) {
                    tracing::warn!(tenant = %info.user_id, error = %message, "replication: tenant sync failed; others continue");
                }
                entry.failures = entry.failures.saturating_add(1);
                entry.retry_at = Some(Instant::now() + backoff(interval, entry.failures));
                entry.last_error = Some(message);
            }
        }
    }
    Ok(consumed)
}

/// The follower loop: idle until something is followed, then poll the leader
/// every interval, backing off while the leader cannot be reached. Runs for the
/// life of the server.
pub async fn run_follower(manager: Arc<MultiUserMemoryManager>) {
    let interval = interval_from_env();
    let http = match reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(error = %e, "replication: no HTTP client; follower disabled");
            return;
        }
    };
    let mut failures = 0u32;
    loop {
        let (target, generation) = {
            let status = FOLLOW.read();
            (status.target.clone(), status.generation)
        };
        let mut wait = interval;
        if let Some(target) = target {
            let outcome = poll_once(&manager, &http, &target, generation, interval).await;
            let mut status = FOLLOW.write();
            if status.generation == generation {
                status.polls += 1;
                status.last_poll = Some(Utc::now());
                match outcome {
                    Ok(consumed) => {
                        if status.last_error.take().is_some() {
                            tracing::info!(source = %target.source, "replication: leader reachable again");
                        }
                        failures = 0;
                        if consumed > 0 {
                            tracing::info!(source = %target.source, consumed, "replication: applied frames");
                        }
                    }
                    Err(e) => {
                        let message = format!("{e:#}");
                        if status.last_error.as_deref() != Some(message.as_str()) {
                            tracing::warn!(source = %target.source, error = %message, "replication: leader unreachable; backing off");
                        }
                        status.last_error = Some(message);
                        failures = failures.saturating_add(1);
                        wait = backoff(interval, failures);
                    }
                }
            } else {
                failures = 0;
            }
        }
        tokio::time::sleep(wait).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_path_is_one_file_per_source() {
        let base = Path::new("/data");
        assert_eq!(
            cursor_path(base, "http://127.0.0.1:3030"),
            PathBuf::from("/data/replication/127-0-0-1-3030.json")
        );
        assert_ne!(
            cursor_path(base, "http://127.0.0.1:3030"),
            cursor_path(base, "http://127.0.0.1:3031")
        );
    }

    #[test]
    fn cursor_round_trips() {
        let dir = std::env::temp_dir().join(format!("veld-repl-cursor-{}", std::process::id()));
        let path = cursor_path(&dir, "http://leader:3030");
        let mut users = BTreeMap::new();
        users.insert(
            "alice".to_string(),
            UserProgress {
                applied_lsn: Some(41),
                applied_records: 42,
                journaled_records: 7,
                source_bytes: 1234,
                leader_head_lsn: Some(90),
                log_created: Some(1_700_000_000),
                diverged: Some("rolled back".into()),
                last_error: None,
                failures: 2,
                retry_at: Some(Instant::now()),
            },
        );
        save_cursor(&path, &users).unwrap();
        let back = load_cursor(&path);
        assert_eq!(back["alice"].applied_lsn, Some(41));
        assert_eq!(back["alice"].source_bytes, 1234);
        assert_eq!(back["alice"].log_created, Some(1_700_000_000));
        assert_eq!(back["alice"].diverged.as_deref(), Some("rolled back"));
        assert!(back["alice"].retry_at.is_none(), "backoff deadlines are not persisted");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cursor_from_before_these_fields_still_loads() {
        let dir = std::env::temp_dir().join(format!("veld-repl-legacy-{}", std::process::id()));
        let path = cursor_path(&dir, "http://leader:3030");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            r#"{"bob":{"applied_lsn":5,"applied_records":6,"source_bytes":700}}"#,
        )
        .unwrap();
        let back = load_cursor(&path);
        assert_eq!(back["bob"].applied_lsn, Some(5));
        assert!(back["bob"].diverged.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unparseable_cursor_is_set_aside_not_silently_reused() {
        let dir = std::env::temp_dir().join(format!("veld-repl-bad-{}", std::process::id()));
        let path = cursor_path(&dir, "http://leader:3030");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{not json").unwrap();
        assert!(load_cursor(&path).is_empty());
        assert!(!path.exists());
        assert!(path.with_extension("json.unreadable").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sources_are_normalized_and_checked() {
        assert_eq!(
            normalize_source_with(" http://127.0.0.1:1/ ", false).unwrap(),
            "http://127.0.0.1:1"
        );
        assert!(normalize_source_with("a:1", false).is_err());
        assert!(normalize_source_with("http://user:pw@127.0.0.1:1", false).is_err());
    }

    #[test]
    fn a_remote_leader_must_be_https_unless_allowed() {
        assert!(normalize_source_with("http://10.0.0.5:3030", false).is_err());
        assert!(normalize_source_with("http://10.0.0.5:3030", true).is_ok());
        assert!(normalize_source_with("https://leader.example:3030", false).is_ok());
        assert!(normalize_source_with("http://localhost:3030", false).is_ok());
        assert!(normalize_source_with("http://[::1]:3030", false).is_ok());
    }

    #[test]
    fn loopback_and_same_instance_detection() {
        assert!(is_loopback_source("http://127.0.0.1:3030"));
        assert!(is_loopback_source("http://localhost:3030"));
        assert!(is_loopback_source("http://[::1]:3030"));
        assert!(!is_loopback_source("https://leader.example:3030"));
        assert!(is_same_instance("http://localhost:3030", "127.0.0.1", 3030));
        assert!(is_same_instance("http://127.0.0.1:3030", "0.0.0.0", 3030));
        assert!(!is_same_instance("http://127.0.0.1:3031", "127.0.0.1", 3030));
        assert_eq!(source_port("https://leader.example"), Some(443));
    }

    #[test]
    fn the_server_key_is_only_offered_to_a_local_leader() {
        assert!(default_api_key_for("https://leader.example:3030").is_none());
    }

    fn page(head_lsn: u64, log_created: u64) -> FeedPage {
        FeedPage {
            user_id: "u".into(),
            records: Vec::new(),
            head_lsn,
            more: false,
            log_created,
        }
    }

    #[test]
    fn divergence_catches_a_leader_that_moved_backwards() {
        let fresh = UserProgress::default();
        assert!(divergence(&fresh, &page(0, 7)).is_none());

        let caught_up = UserProgress {
            applied_lsn: Some(99),
            log_created: Some(7),
            ..UserProgress::default()
        };
        assert!(divergence(&caught_up, &page(100, 7)).is_none());
        assert!(divergence(&caught_up, &page(100, 8)).is_some(), "log recreated");
        assert!(divergence(&caught_up, &page(99, 7)).is_some(), "head at the applied lsn");
        assert!(divergence(&caught_up, &page(40, 7)).is_some(), "rolled back");
        assert!(divergence(&caught_up, &page(100, 0)).is_none(), "legacy log has no identity");
    }

    #[test]
    fn a_stale_poll_cannot_write_into_a_new_target() {
        let mut status = FollowStatus {
            generation: 3,
            ..FollowStatus::default()
        };
        let p = page(10, 1);
        assert!(status.record_page(2, "u", &p, 9, 10, 4).is_none());
        assert!(status.users.is_empty());
        let saved = status.record_page(3, "u", &p, 9, 10, 4).expect("current poll");
        assert_eq!(saved["u"].applied_lsn, Some(9));
        assert_eq!(saved["u"].leader_head_lsn, Some(10));
        assert_eq!(status.pages, 1);
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let i = Duration::from_secs(2);
        assert_eq!(backoff(i, 0), Duration::from_secs(2));
        assert_eq!(backoff(i, 3), Duration::from_secs(16));
        assert_eq!(backoff(i, 30), MAX_BACKOFF);
    }

    #[test]
    fn echo_is_bounded_on_a_char_boundary() {
        let long = "é".repeat(1000);
        let out = echo(&long);
        assert_eq!(out.chars().count(), MAX_ECHO_CHARS + 1);
        assert!(out.ends_with('…'));
        assert_eq!(echo("  short  "), "short");
    }

    fn frame(lsn: u64, id: &str, kind: &str) -> (u64, IntentPayload) {
        frame_for("u", lsn, id, kind)
    }

    fn frame_for(user: &str, lsn: u64, id: &str, kind: &str) -> (u64, IntentPayload) {
        let sv = Some(payload::CURRENT_PAYLOAD_SCHEMA_VERSION);
        let user_id = user.to_string();
        let p = match kind {
            "remember" => IntentPayload::Remember { user_id, memory_id: id.into(), memory_bincode: vec![lsn as u8], schema_version: sv },
            "update" => IntentPayload::Update { user_id, memory_id: id.into(), memory_bincode: vec![lsn as u8], schema_version: sv },
            "forget" => IntentPayload::Forget { user_id, memory_id: id.into(), schema_version: sv },
            _ => IntentPayload::Anchor { user_id, memory_id: id.into(), importance: 0.9, schema_version: sv },
        };
        (lsn, p)
    }

    #[test]
    fn a_page_naming_another_tenant_is_refused() {
        let ok = vec![frame(0, "a", "remember"), frame(1, "b", "forget")];
        assert!(check_tenant("u", &ok).is_ok());
        let mixed = vec![frame(0, "a", "remember"), frame_for("other", 1, "b", "forget")];
        let err = check_tenant("u", &mixed).unwrap_err().to_string();
        assert!(err.contains("other"));
    }

    #[test]
    fn coalesce_keeps_the_last_snapshot_and_what_follows_it() {
        let page = vec![
            frame(0, "a", "remember"),
            frame(1, "b", "remember"),
            frame(2, "a", "update"),
            frame(3, "a", "anchor"),
            frame(4, "a", "update"),
            frame(5, "b", "forget"),
            frame(6, "c", "anchor"),
        ];
        let out: Vec<u64> = coalesce(page).into_iter().map(|(l, _)| l).collect();
        // a: last snapshot is lsn 4; b: snapshot 1 then its forget; c: an anchor alone.
        assert_eq!(out, vec![4, 1, 5, 6]);
    }

    #[test]
    fn coalesce_lets_a_later_remember_undo_a_forget() {
        let page = vec![frame(0, "a", "remember"), frame(1, "a", "forget"), frame(2, "a", "remember")];
        let out: Vec<u64> = coalesce(page).into_iter().map(|(l, _)| l).collect();
        assert_eq!(out, vec![2]);
    }

    #[test]
    fn feed_frames_round_trip_through_base64() {
        let payload = IntentPayload::Forget {
            user_id: "u".into(),
            memory_id: "m".into(),
            schema_version: Some(payload::CURRENT_PAYLOAD_SCHEMA_VERSION),
        };
        let bytes = payload::encode(&payload).unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let back = payload::decode(&base64::engine::general_purpose::STANDARD.decode(b64).unwrap()).unwrap();
        assert_eq!(back.memory_id(), "m");
    }
}
