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
//!   sealed again by the follower's own keystore on append. The admin key gates
//!   the feed; the transport is whatever the two base URLs are.
//!
//! Change detection is by the byte length of each user's `intent.log`, read from
//! the filesystem without opening the user's store, so an idle poll costs one
//! `stat` per user on the leader.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::handlers::state::MultiUserMemoryManager;
use crate::intent_log::{payload, IntentPayload};
use crate::memory::types::MemoryId;
use crate::memory::ForgetCriteria;

pub const ENV_SOURCE: &str = "VELD_REPLICATE_FROM";
pub const ENV_ADMIN_KEY: &str = "VELD_REPLICATE_ADMIN_KEY";
/// The leader's per-request API key; defaults to this server's own first key,
/// which is right whenever both instances share one `config.toml`.
pub const ENV_API_KEY: &str = "VELD_REPLICATE_API_KEY";
pub const ENV_INTERVAL_SECS: &str = "VELD_REPLICATE_INTERVAL_SECS";
pub const DEFAULT_INTERVAL: Duration = Duration::from_secs(2);
/// Frames per page of the feed; a page is one HTTP round trip.
pub const FEED_LIMIT: usize = 500;
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
/// middleware reads; the follower's default for the leader.
pub fn own_api_key() -> Option<String> {
    ["VELD_API_KEYS", "VELD_API_KEY", "VELD_DEV_API_KEY"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .flat_map(|v| v.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>())
        .find(|k| !k.is_empty())
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
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FollowStatus {
    pub target: Option<FollowTarget>,
    pub polls: u64,
    pub last_poll: Option<DateTime<Utc>>,
    pub last_applied: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
    pub users: BTreeMap<String, UserProgress>,
}

static FOLLOW: LazyLock<RwLock<FollowStatus>> = LazyLock::new(|| RwLock::new(FollowStatus::default()));

pub fn status() -> FollowStatus {
    FOLLOW.read().clone()
}

pub fn normalize_source(raw: &str) -> Result<String> {
    let s = raw.trim().trim_end_matches('/');
    if !(s.starts_with("http://") || s.starts_with("https://")) {
        return Err(anyhow!("source must be an http(s) URL, got '{raw}'"));
    }
    Ok(s.to_string())
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

/// Leader side: frames of `user_id` after `after_lsn`, at most `limit`. Blocking.
pub fn feed(
    manager: &MultiUserMemoryManager,
    user_id: &str,
    after_lsn: Option<u64>,
    limit: usize,
) -> Result<FeedPage> {
    let writer = manager.get_user_journaled_writer(user_id)?;
    let (iter, head_lsn) = {
        let guard = writer.lock();
        (guard.log().iter()?, guard.log().next_lsn().0)
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
    })
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

/// Follower side: apply a page of leader frames to the canonical store, then
/// journal each applied frame here so the local projections and log follow. The
/// page is coalesced first; snapshots are indexed with one BM25 commit. Idempotent:
/// a snapshot is an upsert, a forget of an unknown id is a no-op. Returns the
/// number of frames journaled. Blocking.
pub fn apply_replicated_page(
    manager: &MultiUserMemoryManager,
    user_id: &str,
    frames: Vec<(u64, IntentPayload)>,
) -> Result<u64> {
    let frames = coalesce(frames);
    if frames.is_empty() {
        return Ok(0);
    }
    let memory_sys = manager.get_user_memory(user_id)?;
    let mut snapshots: Vec<crate::memory::types::Memory> = Vec::new();
    for (lsn, payload) in &frames {
        if let IntentPayload::Remember { memory_bincode, .. }
        | IntentPayload::Update { memory_bincode, .. } = payload
        {
            // The snapshot ages like the store's own records: frames journaled before a
            // layout change need the legacy decode arms, the same ones the projections use.
            let memory = crate::memory::storage::decode_memory_snapshot(memory_bincode)
                .map_err(|e| anyhow!("decoding the memory snapshot at lsn {lsn}: {e}"))?;
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

fn cursor_path(base: &std::path::Path, source: &str) -> PathBuf {
    let slug: String = source
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    base.join(CURSOR_DIR).join(format!("{slug}.json"))
}

fn load_cursor(path: &std::path::Path) -> BTreeMap<String, UserProgress> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

fn save_cursor(path: &std::path::Path, users: &BTreeMap<String, UserProgress>) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(users)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Start (or switch) following `source`. The cursor saved for that source is
/// resumed, so a follower that was stopped picks up where it left off.
pub fn follow(
    base: &std::path::Path,
    source: &str,
    admin_key: &str,
    api_key: Option<&str>,
) -> Result<FollowStatus> {
    let source = normalize_source(source)?;
    let api_key = api_key
        .map(str::to_string)
        .filter(|k| !k.trim().is_empty())
        .or_else(own_api_key)
        .ok_or_else(|| anyhow!("no API key for the leader: pass one, or set {ENV_API_KEY}"))?;
    let users = load_cursor(&cursor_path(base, &source));
    let mut status = FOLLOW.write();
    *status = FollowStatus {
        target: Some(FollowTarget {
            source,
            admin_key: admin_key.to_string(),
            api_key,
            since: Utc::now(),
        }),
        users,
        ..FollowStatus::default()
    };
    Ok(status.clone())
}

pub fn unfollow() -> FollowStatus {
    let mut status = FOLLOW.write();
    status.target = None;
    status.clone()
}

/// `VELD_REPLICATE_FROM` + `VELD_REPLICATE_ADMIN_KEY` at boot.
pub fn follow_from_env(base: &std::path::Path) -> Result<Option<FollowStatus>> {
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
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!("GET {url}: {status}: {}", text.trim()));
    }
    serde_json::from_str(&text).with_context(|| format!("GET {url}: decoding the reply"))
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

async fn poll_once(
    manager: &Arc<MultiUserMemoryManager>,
    http: &reqwest::Client,
    target: &FollowTarget,
) -> Result<(u64, bool)> {
    let leader: LeaderStatus = get_json(http, target, "/api/admin/replication/status").await?;
    let cursor = cursor_path(&manager.base_path, &target.source);
    let mut applied_total = 0u64;
    for info in leader.users {
        let progress = FOLLOW
            .read()
            .users
            .get(&info.user_id)
            .cloned()
            .unwrap_or_default();
        if progress.applied_lsn.is_some() && progress.source_bytes == info.log_bytes {
            continue;
        }
        let mut after = progress.applied_lsn;
        loop {
            let path = format!(
                "/api/admin/replication/records?user_id={}&after_lsn={}&limit={FEED_LIMIT}",
                percent_encode(&info.user_id),
                after.map(|l| l.to_string()).unwrap_or_default()
            );
            let page: FeedPage = get_json(http, target, &path).await?;
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
                    .with_context(|| {
                        format!("applying lsns up to {last_lsn} of {}", info.user_id)
                    })?;
            after = Some(last_lsn);
            applied_total += received;
            let mut status = FOLLOW.write();
            let entry = status.users.entry(info.user_id.clone()).or_default();
            entry.applied_lsn = after;
            entry.applied_records += received;
            entry.journaled_records += journaled;
            status.last_applied = Some(Utc::now());
            drop(status);
            if !page.more {
                break;
            }
        }
        let users = {
            let mut status = FOLLOW.write();
            let entry = status.users.entry(info.user_id.clone()).or_default();
            entry.source_bytes = info.log_bytes;
            if entry.applied_lsn.is_none() {
                entry.applied_lsn = after;
            }
            status.users.clone()
        };
        save_cursor(&cursor, &users)?;
    }
    Ok((applied_total, true))
}

/// The follower loop: idle until something is followed, then poll the leader
/// every interval. Runs for the life of the server.
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
    loop {
        let target = FOLLOW.read().target.clone();
        if let Some(target) = target {
            let outcome = poll_once(&manager, &http, &target).await;
            let mut status = FOLLOW.write();
            if status.target.as_ref().map(|t| &t.source) == Some(&target.source) {
                status.polls += 1;
                status.last_poll = Some(Utc::now());
                match outcome {
                    Ok((applied, _)) => {
                        status.last_error = None;
                        if applied > 0 {
                            tracing::info!(source = %target.source, applied, "replication: applied frames");
                        }
                    }
                    Err(e) => {
                        tracing::warn!(source = %target.source, error = %e, "replication: poll failed");
                        status.last_error = Some(format!("{e:#}"));
                    }
                }
            }
        }
        tokio::time::sleep(interval).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_path_is_one_file_per_source() {
        let base = std::path::Path::new("/data");
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
            },
        );
        save_cursor(&path, &users).unwrap();
        let back = load_cursor(&path);
        assert_eq!(back["alice"].applied_lsn, Some(41));
        assert_eq!(back["alice"].source_bytes, 1234);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sources_are_normalized_and_checked() {
        assert_eq!(normalize_source(" http://a:1/ ").unwrap(), "http://a:1");
        assert!(normalize_source("a:1").is_err());
    }

    fn frame(lsn: u64, id: &str, kind: &str) -> (u64, IntentPayload) {
        let sv = Some(payload::CURRENT_PAYLOAD_SCHEMA_VERSION);
        let p = match kind {
            "remember" => IntentPayload::Remember { user_id: "u".into(), memory_id: id.into(), memory_bincode: vec![lsn as u8], schema_version: sv },
            "update" => IntentPayload::Update { user_id: "u".into(), memory_id: id.into(), memory_bincode: vec![lsn as u8], schema_version: sv },
            "forget" => IntentPayload::Forget { user_id: "u".into(), memory_id: id.into(), schema_version: sv },
            _ => IntentPayload::Anchor { user_id: "u".into(), memory_id: id.into(), importance: 0.9, schema_version: sv },
        };
        (lsn, p)
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
