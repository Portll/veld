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
use tokio_util::sync::CancellationToken;

use crate::embeddings::space::{SlotSpaces, SpaceId};
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
/// Path to a PEM bundle of extra trust roots, added to the built-in webpki roots,
/// for a leader whose certificate a private CA signed.
pub const ENV_CA_CERT: &str = "VELD_CA_CERT";
/// The follower's client certificate chain (PEM, leaf first) and its private key
/// (PEM), presented to a leader whose TLS listener requires one. Both or neither.
pub const ENV_CLIENT_CERT: &str = "VELD_REPLICATE_CLIENT_CERT";
pub const ENV_CLIENT_KEY: &str = "VELD_REPLICATE_CLIENT_KEY";
pub const DEFAULT_INTERVAL: Duration = Duration::from_secs(2);
/// Per-request timeout of the client that polls the leader.
const POLL_TIMEOUT: Duration = Duration::from_secs(60);
/// Per-request timeout of the follow route's check that the leader does not follow us.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
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
    /// How the leader reads a vector slot that records no space. A snapshot journaled before
    /// spaces were recorded carries such slots, and a follower with another embedder must not
    /// read them as its own. Absent from a leader that predates spaces: the follower then reads
    /// them as its own, as it always did.
    #[serde(default)]
    pub slot_spaces: Option<SlotSpaces>,
    /// The spaces of the vectors in this page's snapshots, as the leader reads them.
    #[serde(default)]
    pub spaces: Vec<SpaceId>,
}

/// What a follower follows. The keys never leave the process.
#[derive(Debug, Clone, Serialize)]
pub struct FollowTarget {
    pub source: String,
    #[serde(skip)]
    pub admin_key: String,
    #[serde(skip)]
    pub api_key: String,
    /// Built by [`follow`] from the TLS variables as they stood then.
    #[serde(skip)]
    pub http: reqwest::Client,
    pub since: DateTime<Utc>,
}

/// This server's own first API key, from the same variables the auth
/// middleware reads.
pub fn own_api_key() -> Option<String> {
    ["VELD_API_KEYS", "VELD_API_KEY", "VELD_DEV_API_KEY"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .flat_map(|v| {
            v.split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>()
        })
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
    /// The leader's next LSN when this tenant was last read.
    #[serde(default)]
    pub leader_head_lsn: Option<u64>,
    /// Leader frames not yet applied here: `leader_head_lsn - 1 - applied_lsn`, or the
    /// whole leader log when nothing is applied.
    #[serde(default)]
    pub lag: Option<u64>,
    /// The leader log identity this cursor belongs to (see [`FeedPage::log_created`]).
    #[serde(default)]
    pub log_created: Option<u64>,
    /// The spaces of the vectors in the leader's last page (see [`FeedPage::spaces`]).
    #[serde(default)]
    pub leader_spaces: Vec<SpaceId>,
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

impl UserProgress {
    fn refresh_lag(&mut self) {
        self.lag = match (self.leader_head_lsn, self.applied_lsn) {
            (Some(head), Some(applied)) => Some(head.saturating_sub(1).saturating_sub(applied)),
            (Some(head), None) => Some(head),
            (None, _) => None,
        };
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FollowStatus {
    pub target: Option<FollowTarget>,
    /// Bumped by every follow and unfollow; a poll started under an older value
    /// stops at its next page and writes nothing.
    pub generation: u64,
    /// Passes over every tenant that finished. An initial sync is one pass that can
    /// run for hours: `pass_started`, `syncing` and `pages` show it moving.
    pub polls: u64,
    pub pages: u64,
    /// When the pass in progress, or the last one, began.
    pub pass_started: Option<DateTime<Utc>>,
    /// The tenant the pass in progress is syncing.
    pub syncing: Option<String>,
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
        if !page.spaces.is_empty() {
            entry.leader_spaces = page.spaces.clone();
        }
        entry.refresh_lag();
        self.pages += 1;
        self.last_applied = Some(Utc::now());
        Some(self.users.clone())
    }
}

static FOLLOW: LazyLock<RwLock<FollowStatus>> =
    LazyLock::new(|| RwLock::new(FollowStatus::default()));

static INSTANCE_ID: LazyLock<uuid::Uuid> = LazyLock::new(uuid::Uuid::new_v4);

/// This process's identity, new at every start. The status route reports it, so the
/// follow route recognises this process behind any URL: the TLS listener, a DNS name,
/// a proxy.
pub fn instance_id() -> uuid::Uuid {
    *INSTANCE_ID
}

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

/// Extra trust roots and an optional client identity for reaching a leader.
#[derive(Debug, Default)]
struct LeaderTls {
    roots: Vec<reqwest::Certificate>,
    identity: Option<reqwest::Identity>,
    /// `NAME=path` for each file in use, so a client that will not build names them.
    files: Vec<String>,
}

fn read_named(var: &str, path: &str) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("{var}={path}: cannot read the file"))
}

fn pem_certificates(var: &str, path: &str, pem: &[u8]) -> Result<Vec<reqwest::Certificate>> {
    let certs = reqwest::Certificate::from_pem_bundle(pem)
        .with_context(|| format!("{var}={path}: not a PEM certificate bundle"))?;
    if certs.is_empty() {
        bail!("{var}={path}: no PEM certificate in the file");
    }
    Ok(certs)
}

impl LeaderTls {
    fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let get = |var: &str| {
            get(var)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let mut tls = Self::default();
        if let Some(path) = get(ENV_CA_CERT) {
            tls.roots = pem_certificates(ENV_CA_CERT, &path, &read_named(ENV_CA_CERT, &path)?)?;
            tls.files.push(format!("{ENV_CA_CERT}={path}"));
        }
        let (cert, key) = match (get(ENV_CLIENT_CERT), get(ENV_CLIENT_KEY)) {
            (None, None) => return Ok(tls),
            (Some(_), None) => bail!(
                "{ENV_CLIENT_CERT} is set but {ENV_CLIENT_KEY} is not: a client identity needs both"
            ),
            (None, Some(_)) => bail!(
                "{ENV_CLIENT_KEY} is set but {ENV_CLIENT_CERT} is not: a client identity needs both"
            ),
            (Some(cert), Some(key)) => (cert, key),
        };
        let mut pem = read_named(ENV_CLIENT_CERT, &cert)?;
        pem_certificates(ENV_CLIENT_CERT, &cert, &pem)?;
        pem.push(b'\n');
        pem.extend(read_named(ENV_CLIENT_KEY, &key)?);
        let identity = reqwest::Identity::from_pem(&pem).with_context(|| {
            format!(
                "{ENV_CLIENT_CERT}={cert} with {ENV_CLIENT_KEY}={key}: \
                 not a PEM certificate chain and private key"
            )
        })?;
        tls.identity = Some(identity);
        tls.files.push(format!("{ENV_CLIENT_CERT}={cert}"));
        tls.files.push(format!("{ENV_CLIENT_KEY}={key}"));
        Ok(tls)
    }

    fn client(self, timeout: Duration) -> Result<reqwest::Client> {
        let mut builder = reqwest::Client::builder().timeout(timeout);
        for root in self.roots {
            builder = builder.add_root_certificate(root);
        }
        if let Some(identity) = self.identity {
            builder = builder.identity(identity);
        }
        builder.build().with_context(|| {
            if self.files.is_empty() {
                "building the HTTP client".to_string()
            } else {
                format!("building the HTTP client with {}", self.files.join(", "))
            }
        })
    }
}

/// The client a follower reaches its leader with: the built-in webpki roots plus
/// any in [`ENV_CA_CERT`], presenting [`ENV_CLIENT_CERT`] and [`ENV_CLIENT_KEY`]
/// when both are set. The files are read on every call, so following again after a
/// certificate rotation picks up the new ones.
pub fn leader_client(timeout: Duration) -> Result<reqwest::Client> {
    LeaderTls::from_lookup(|var| std::env::var(var).ok())?.client(timeout)
}

/// Leader side: every user's log length, by `stat` alone.
pub fn user_logs(manager: &MultiUserMemoryManager) -> Vec<UserLogInfo> {
    // Only tenants with a log: `records` answers 404 for the others, so listing them
    // would leave a follower retrying a tenant it can never be fed.
    let mut out: Vec<UserLogInfo> = manager
        .list_users()
        .into_iter()
        .filter_map(|user_id| {
            let head = manager.user_dir(&user_id).ok()?.join("intent.log");
            head.is_file().then(|| UserLogInfo {
                log_bytes: crate::intent_log::log_bytes(&head).unwrap_or(0),
                user_id,
            })
        })
        .collect();
    out.sort_by(|a, b| a.user_id.cmp(&b.user_id));
    out
}

/// Leader side: the lowest cursor each tenant's feed was asked to read past, by
/// hour, so a journal purge can leave alone the `Forget` frames a follower has
/// not read yet. Followers are not named; a lagging one shows as a low cursor,
/// and its low hours drop out as the purge window moves past them. Persisted
/// beside the followers' own cursors, so a restart does not forget a follower
/// that has not polled since.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FeedFloors {
    /// Tenant -> unix hour -> lowest `after_lsn` asked for in that hour.
    tenants: BTreeMap<String, BTreeMap<u64, u64>>,
    #[serde(skip)]
    path: Option<PathBuf>,
}

impl FeedFloors {
    const FILE: &'static str = "feed-floors.json";

    /// Load the floors kept under `base`; a missing or unreadable file starts
    /// empty (logged), as the follower cursors do.
    pub fn load(base: &Path) -> Self {
        let path = base.join(CURSOR_DIR).join(Self::FILE);
        let mut floors = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str::<FeedFloors>(&text).unwrap_or_else(|e| {
                tracing::warn!(path = %path.display(), error = %e, "replication feed floors unreadable; starting empty");
                FeedFloors::default()
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => FeedFloors::default(),
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "replication feed floors unreadable; starting empty");
                FeedFloors::default()
            }
        };
        floors.path = Some(path);
        floors
    }

    fn hour(at: std::time::SystemTime) -> u64 {
        at.duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() / 3600)
            .unwrap_or(0)
    }

    /// Note a feed page asked for after `after_lsn`, dropping hours older than
    /// `window`. Written to disk only when it changes what the floor can be: a
    /// new hour, or a lower cursor within one.
    pub fn note(
        &mut self,
        user_id: &str,
        after_lsn: u64,
        window: Duration,
        at: std::time::SystemTime,
    ) {
        let hour = Self::hour(at);
        let oldest = hour.saturating_sub(window.as_secs() / 3600);
        let hours = self.tenants.entry(user_id.to_string()).or_default();
        let before = hours.len();
        hours.retain(|h, _| *h >= oldest);
        let changed = match hours.get_mut(&hour) {
            Some(low) if *low <= after_lsn => hours.len() != before,
            Some(low) => {
                *low = after_lsn;
                true
            }
            None => {
                hours.insert(hour, after_lsn);
                true
            }
        };
        if changed {
            self.save();
        }
    }

    /// The lowest cursor asked for in the last `window`, or `None` when no
    /// follower read this tenant's feed in that time.
    pub fn floor(&self, user_id: &str, window: Duration, at: std::time::SystemTime) -> Option<u64> {
        let oldest = Self::hour(at).saturating_sub(window.as_secs() / 3600);
        self.tenants
            .get(user_id)?
            .range(oldest..)
            .map(|(_, low)| *low)
            .min()
    }

    fn save(&self) {
        let Some(path) = &self.path else {
            return;
        };
        let written = serde_json::to_vec(self)
            .map_err(std::io::Error::other)
            .and_then(|bytes| {
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir)?;
                }
                let tmp = path.with_extension("json.tmp");
                std::fs::write(&tmp, bytes)?;
                std::fs::rename(&tmp, path)
            });
        if let Err(e) = written {
            tracing::warn!(path = %path.display(), error = %e, "replication feed floors not saved");
        }
    }
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
        let floor = after_lsn.map(|l| l + 1).unwrap_or(0);
        // Only from the frame the follower needs: a page used to read the log
        // from its first frame and skip every one below the cursor.
        (
            guard.log().iter_from(crate::intent_log::Lsn(floor))?,
            guard.log().next_lsn().0,
            log_created,
        )
    };
    let floor = after_lsn.map(|l| l + 1).unwrap_or(0);
    let (records, more) = page_below_head(iter, floor, head_lsn, limit)?;
    let slot_spaces = manager.slot_spaces(user_id);
    let spaces = page_spaces(&records, slot_spaces.as_ref());
    Ok(FeedPage {
        user_id: user_id.to_string(),
        records,
        head_lsn,
        more,
        log_created,
        slot_spaces,
        spaces,
    })
}

/// The spaces of the vectors the snapshots in `records` hold, unrecorded slots read with
/// `slot_spaces`, sorted. A frame that does not decode names none: the follower reports it.
fn page_spaces(records: &[FeedRecord], slot_spaces: Option<&SlotSpaces>) -> Vec<SpaceId> {
    let primary = slot_spaces.map(|s| &s.primary);
    let secondary = slot_spaces.and_then(|s| s.secondary.as_ref());
    let mut spaces = std::collections::BTreeSet::new();
    for record in records {
        let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&record.frame_b64) else {
            continue;
        };
        let Ok(
            IntentPayload::Remember { memory_bincode, .. }
            | IntentPayload::Update { memory_bincode, .. },
        ) = payload::decode(&bytes)
        else {
            continue;
        };
        let Ok(memory) = crate::memory::storage::decode_memory_snapshot(&memory_bincode) else {
            continue;
        };
        let experience = &memory.experience;
        spaces.extend(experience.primary_space(primary));
        spaces.extend(experience.secondary_space(secondary));
        spaces.extend(experience.other_vectors.iter().map(|v| v.space.clone()));
    }
    spaces.into_iter().collect()
}

/// Frames from `floor` up to `limit`, ending at the last frame below `head_lsn`.
/// The file is read after the writer lock is released, so anything at or past the
/// head was appended later: it may still be partly flushed (and read as torn) or be
/// rolled back by a failed sync, after which its LSN names a different frame. LSNs
/// are contiguous, so the frame at `head_lsn - 1` is the last one read.
fn page_below_head<E>(
    frames: impl Iterator<Item = std::result::Result<crate::intent_log::IntentRecord, E>>,
    floor: u64,
    head_lsn: u64,
    limit: usize,
) -> Result<(Vec<FeedRecord>, bool)>
where
    E: std::error::Error + Send + Sync + 'static,
{
    let mut records = Vec::new();
    if floor >= head_lsn {
        return Ok((records, false));
    }
    for frame in frames {
        let record = frame.context("reading the intent log")?;
        if record.lsn.0 >= head_lsn {
            break;
        }
        if record.lsn.0 >= floor {
            if records.len() >= limit {
                return Ok((records, true));
            }
            records.push(FeedRecord {
                lsn: record.lsn.0,
                frame_b64: base64::engine::general_purpose::STANDARD.encode(&record.payload),
            });
        }
        if record.lsn.0 + 1 == head_lsn {
            break;
        }
    }
    Ok((records, false))
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
            matches!(
                p,
                IntentPayload::Remember { .. } | IntentPayload::Update { .. }
            )
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
/// journal the applied frames here, as one batch, so the local projections and log follow. The
/// page is coalesced first; the BM25 projection indexes snapshots as their frames
/// are journaled, on its own commit cadence. Idempotent:
/// a snapshot is an upsert, a forget of an unknown id is a no-op. Returns the
/// number of frames journaled. Blocking.
///
/// Each snapshot's vectors are read as the leader reads them (`leader`, from the page) and
/// kept in this corpus's slots only when they are in its spaces; a vector in another space is
/// kept beside them, and the memory is queued for its vectors here. The frame journaled here is
/// the snapshot as this corpus stores it, so the local projections never read the leader's
/// vectors as this corpus's.
pub fn apply_replicated_page(
    manager: &MultiUserMemoryManager,
    user_id: &str,
    frames: Vec<(u64, IntentPayload)>,
    leader: Option<&SlotSpaces>,
) -> Result<u64> {
    check_tenant(user_id, &frames)?;
    let mut frames = coalesce(frames);
    if frames.is_empty() {
        return Ok(0);
    }
    let memory_sys = manager.get_user_memory(user_id)?;
    let mut snapshots: Vec<crate::memory::types::Memory> = Vec::new();
    let mut arrived: BTreeMap<SpaceId, usize> = BTreeMap::new();
    for (lsn, payload) in &mut frames {
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
            let mut memory = crate::memory::storage::decode_memory_snapshot(memory_bincode)
                .map_err(|e| anyhow!("decoding the memory snapshot at lsn {lsn}: {e}"))?;
            if memory.id.0.to_string() != *memory_id {
                bail!(
                    "frame lsn {lsn} names memory {memory_id} but its snapshot is memory {}",
                    memory.id.0
                );
            }
            if memory_sys.read().normalise_replicated(&mut memory, leader) {
                *memory_bincode =
                    bincode::serde::encode_to_vec(&memory, bincode::config::standard())
                        .map_err(|e| anyhow!("re-encoding the snapshot at lsn {lsn}: {e}"))?;
            }
            for kept in &memory.experience.other_vectors {
                *arrived.entry(kept.space.clone()).or_insert(0) += 1;
            }
            snapshots.push(memory);
        }
    }
    memory_sys.read().upsert_memories(&snapshots)?;
    for (_, payload) in &frames {
        apply_non_snapshot(manager, user_id, payload)?;
    }
    // One writer lock, append batch and fsync for the page rather than one per frame.
    // All or nothing: a page that fails journals no frame, and its retry adds no
    // duplicates of the frames before the failure.
    let payloads: Vec<IntentPayload> = frames.into_iter().map(|(_, payload)| payload).collect();
    let journaled = manager.journal_and_apply_batch(user_id, &payloads)?.len() as u64;
    if !arrived.is_empty() {
        memory_sys.read().note_replicated_spaces(&arrived);
    }
    Ok(journaled)
}

/// The store-side effect of a `Forget`, `Anchor` or `SetAnchored` frame; snapshots are handled in
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
        IntentPayload::SetAnchored {
            memory_id,
            anchored,
            importance,
            ..
        } => {
            let id = MemoryId(uuid::Uuid::parse_str(memory_id)?);
            let guard = memory_sys.read();
            if let Ok(memory) = guard.get_memory(&id) {
                memory.set_anchored(*anchored);
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
/// running for the previous target stops at its next page. Fails, naming the
/// variable and the file, when the TLS variables do not yield a client.
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
    let http = leader_client(POLL_TIMEOUT)?;
    let users = load_cursor(&cursor_path(base, &source));
    let mut status = FOLLOW.write();
    let generation = status.generation + 1;
    *status = FollowStatus {
        target: Some(FollowTarget {
            source,
            admin_key: admin_key.to_string(),
            api_key,
            http,
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

async fn get_json<T: serde::de::DeserializeOwned>(target: &FollowTarget, path: &str) -> Result<T> {
    let url = format!("{}{}", target.source, path);
    let resp = target
        .http
        .get(&url)
        .header("X-API-Key", &target.api_key)
        .header(
            crate::handlers::admin::ADMIN_API_KEY_HEADER,
            &target.admin_key,
        )
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    let body = read_capped(resp, MAX_REPLY_BYTES)
        .await
        .with_context(|| format!("GET {url}"))?;
    if !status.is_success() {
        bail!(
            "GET {url}: {status}: {}",
            echo(&String::from_utf8_lossy(&body))
        );
    }
    serde_json::from_slice(&body).with_context(|| format!("GET {url}: decoding the reply"))
}

/// The leader's own view: its users' logs, which process it is and whom it follows,
/// from its status route.
#[derive(Debug, Deserialize)]
struct LeaderStatus {
    users: Vec<UserLogInfo>,
    /// Absent from a leader built before the status route reported it.
    #[serde(default)]
    instance_id: Option<uuid::Uuid>,
    #[serde(default)]
    following: Option<serde_json::Value>,
}

/// What a prospective leader says about itself, read before following it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaderIdentity {
    /// `None` from a leader too old to report one.
    pub instance_id: Option<uuid::Uuid>,
    /// The source the leader follows, if any.
    pub follows: Option<String>,
}

impl From<LeaderStatus> for LeaderIdentity {
    fn from(leader: LeaderStatus) -> Self {
        Self {
            instance_id: leader.instance_id,
            follows: leader
                .following
                .and_then(|f| f.get("target")?.get("source")?.as_str().map(str::to_string)),
        }
    }
}

/// Why following a leader would echo frames forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FollowLoop {
    /// The leader is this process, reached by a URL the address check does not know.
    Itself,
    /// The leader already follows this instance.
    Mutual,
}

/// Whether following `leader` would loop, for an instance whose id is `own_id` and
/// whose base URL is `own_base_url`. A leader that reports no id is not taken for
/// this process.
pub fn follow_loop(
    leader: &LeaderIdentity,
    own_id: uuid::Uuid,
    own_base_url: &str,
) -> Option<FollowLoop> {
    if leader.instance_id == Some(own_id) {
        return Some(FollowLoop::Itself);
    }
    (leader.follows.as_deref() == Some(own_base_url)).then_some(FollowLoop::Mutual)
}

/// Which process the leader is and what it follows: used to refuse a loop. `http`
/// comes from [`leader_client`], so the caller can tell a TLS configuration fault
/// from an unreachable leader.
pub async fn probe_leader(
    http: reqwest::Client,
    source: &str,
    admin_key: &str,
    api_key: &str,
) -> Result<LeaderIdentity> {
    let target = FollowTarget {
        source: normalize_source(source)?,
        admin_key: admin_key.to_string(),
        api_key: api_key.to_string(),
        http,
        since: Utc::now(),
    };
    let leader: LeaderStatus = get_json(&target, "/api/admin/replication/status").await?;
    Ok(leader.into())
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
    target: &FollowTarget,
    generation: u64,
    cursor: &Path,
    info: &UserLogInfo,
    shutdown: &CancellationToken,
) -> Result<u64> {
    let mut consumed = 0u64;
    let mut head: u64;
    loop {
        if shutdown.is_cancelled() || !is_current(generation) {
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
            progress
                .applied_lsn
                .map(|l| l.to_string())
                .unwrap_or_default()
        );
        let page: FeedPage = tokio::select! {
            page = get_json(target, &path) => page?,
            () = shutdown.cancelled() => return Ok(consumed),
        };
        head = page.head_lsn;
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
        let leader = page.slot_spaces.clone();
        let journaled = tokio::task::spawn_blocking(move || {
            apply_replicated_page(&m, &user_id, frames, leader.as_ref())
        })
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
        entry.leader_head_lsn = Some(head);
        entry.refresh_lag();
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
    target: &FollowTarget,
    generation: u64,
    interval: Duration,
    shutdown: &CancellationToken,
) -> Result<u64> {
    let leader: LeaderStatus = tokio::select! {
        leader = get_json(target, "/api/admin/replication/status") => leader?,
        () = shutdown.cancelled() => return Ok(0),
    };
    if leader.instance_id == Some(instance_id()) {
        let mut status = FOLLOW.write();
        if status.generation == generation {
            let message = format!("{} is this instance; following stopped", target.source);
            tracing::error!(source = %target.source, "replication: {message}");
            status.target = None;
            status.generation += 1;
            status.last_error = Some(message);
        }
        return Ok(0);
    }
    let cursor = cursor_path(&manager.base_path, &target.source);
    let set_syncing = |tenant: Option<&str>| {
        let mut status = FOLLOW.write();
        if status.generation == generation {
            status.syncing = tenant.map(str::to_string);
        }
    };
    {
        let mut status = FOLLOW.write();
        if status.generation == generation {
            status.pass_started = Some(Utc::now());
        }
    }
    let mut consumed = 0u64;
    for info in &leader.users {
        if shutdown.is_cancelled() || !is_current(generation) {
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
        set_syncing(Some(&info.user_id));
        match sync_user(manager, target, generation, &cursor, info, shutdown).await {
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
    set_syncing(None);
    Ok(consumed)
}

/// The follower loop: idle until something is followed, then poll the leader
/// every interval, backing off while the leader cannot be reached. Returns once
/// `shutdown` is cancelled: a page already being applied finishes and saves its
/// cursor, and no request or page starts after the cancel, so the store flush and
/// the vector-index save at shutdown see no later writes.
pub async fn run_follower(manager: Arc<MultiUserMemoryManager>, shutdown: CancellationToken) {
    let interval = interval_from_env();
    let mut failures = 0u32;
    loop {
        let (target, generation) = {
            let status = FOLLOW.read();
            (status.target.clone(), status.generation)
        };
        let mut wait = interval;
        if let Some(target) = target {
            let outcome = poll_once(&manager, &target, generation, interval, &shutdown).await;
            if shutdown.is_cancelled() {
                break;
            }
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
        tokio::select! {
            () = tokio::time::sleep(wait) => {}
            () = shutdown.cancelled() => break,
        }
    }
    tracing::info!("replication: follower stopped for shutdown");
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
                lag: Some(48),
                log_created: Some(1_700_000_000),
                leader_spaces: vec![SpaceId::new("nomic-embed-text-v1.5", 768)],
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
        assert_eq!(
            back["alice"].leader_spaces,
            vec![SpaceId::new("nomic-embed-text-v1.5", 768)]
        );
        assert_eq!(back["alice"].diverged.as_deref(), Some("rolled back"));
        assert!(
            back["alice"].retry_at.is_none(),
            "backoff deadlines are not persisted"
        );
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
        assert!(!is_same_instance(
            "http://127.0.0.1:3031",
            "127.0.0.1",
            3030
        ));
        assert_eq!(source_port("https://leader.example"), Some(443));
    }

    #[test]
    fn the_server_key_is_only_offered_to_a_local_leader() {
        assert!(default_api_key_for("https://leader.example:3030").is_none());
    }

    #[test]
    fn a_leader_reporting_this_instance_id_is_this_instance_whatever_its_url() {
        let me = uuid::Uuid::new_v4();
        let own_url = "http://127.0.0.1:3030";
        let leader = |instance_id, follows: Option<&str>| LeaderIdentity {
            instance_id,
            follows: follows.map(str::to_string),
        };

        assert_eq!(
            follow_loop(&leader(Some(me), None), me, own_url),
            Some(FollowLoop::Itself)
        );
        assert_eq!(
            follow_loop(&leader(Some(me), Some(own_url)), me, own_url),
            Some(FollowLoop::Itself)
        );
        assert_eq!(
            follow_loop(
                &leader(Some(uuid::Uuid::new_v4()), Some(own_url)),
                me,
                own_url
            ),
            Some(FollowLoop::Mutual)
        );
        assert_eq!(
            follow_loop(&leader(None, Some(own_url)), me, own_url),
            Some(FollowLoop::Mutual)
        );
        assert_eq!(
            follow_loop(
                &leader(Some(uuid::Uuid::new_v4()), Some("http://127.0.0.1:3031")),
                me,
                own_url
            ),
            None
        );
        assert_eq!(
            follow_loop(&leader(None, None), me, own_url),
            None,
            "a leader too old to report an id is followed"
        );
    }

    #[test]
    fn the_leader_status_reports_its_instance_id_and_an_older_one_reads_as_unknown() {
        let id = uuid::Uuid::new_v4();
        let current: LeaderStatus = serde_json::from_value(serde_json::json!({
            "base_url": "http://127.0.0.1:3030",
            "instance_id": id,
            "users": [],
            "following": {"target": {"source": "https://leader.example:3443"}},
        }))
        .unwrap();
        assert_eq!(
            LeaderIdentity::from(current),
            LeaderIdentity {
                instance_id: Some(id),
                follows: Some("https://leader.example:3443".to_string()),
            }
        );

        let older: LeaderStatus = serde_json::from_value(serde_json::json!({
            "base_url": "http://127.0.0.1:3030",
            "users": [],
            "following": null,
        }))
        .unwrap();
        assert_eq!(
            LeaderIdentity::from(older),
            LeaderIdentity {
                instance_id: None,
                follows: None,
            }
        );
    }

    #[test]
    fn the_instance_id_is_fixed_for_the_process() {
        assert_eq!(instance_id(), instance_id());
        assert_eq!(instance_id().get_version_num(), 4);
    }

    fn page(head_lsn: u64, log_created: u64) -> FeedPage {
        FeedPage {
            user_id: "u".into(),
            records: Vec::new(),
            head_lsn,
            more: false,
            log_created,
            slot_spaces: None,
            spaces: Vec::new(),
        }
    }

    #[test]
    fn a_page_from_a_leader_before_spaces_reads_with_no_spaces() {
        let page: FeedPage =
            serde_json::from_str(r#"{"user_id":"u","records":[],"head_lsn":3,"more":false}"#)
                .unwrap();
        assert_eq!(page.slot_spaces, None);
        assert!(page.spaces.is_empty());
    }

    /// A snapshot journaled before spaces were recorded, holding `vector` in its primary slot.
    fn unstamped_snapshot(vector: Vec<f32>) -> (crate::memory::types::Memory, IntentPayload) {
        let memory = crate::memory::types::Memory::new(
            MemoryId(uuid::Uuid::new_v4()),
            crate::memory::types::Experience {
                content: "the lighthouse keeper logs the fog at dawn".to_string(),
                embeddings: Some(vector),
                ..Default::default()
            },
            0.5,
            None,
            None,
            None,
            None,
        );
        let payload = IntentPayload::Remember {
            user_id: "u".into(),
            memory_id: memory.id.0.to_string(),
            memory_bincode: bincode::serde::encode_to_vec(&memory, bincode::config::standard())
                .unwrap(),
            schema_version: Some(payload::CURRENT_PAYLOAD_SCHEMA_VERSION),
        };
        (memory, payload)
    }

    fn unit(dim: usize) -> Vec<f32> {
        vec![1.0 / (dim as f32).sqrt(); dim]
    }

    /// A leader with another embedder of the same width: the follower must not read its
    /// unrecorded vector as its own space's. It keeps it in the leader's space, queues the
    /// memory for a vector of its own, and journals the snapshot as it stores it.
    #[test]
    fn a_leader_vector_in_another_space_is_kept_queued_and_journaled_as_stored() {
        let (manager, _tmp) = crate::handlers::test_helpers::TestHarness::fresh_manager();
        let follower = manager.get_user_memory("u").unwrap();
        let own = follower.read().slot_spaces();
        let leader_space = SpaceId::new("leader-model", own.primary.dimension());
        let vector = unit(own.primary.dimension());
        let (memory, frame) = unstamped_snapshot(vector.clone());
        let leader = SlotSpaces {
            primary: leader_space.clone(),
            secondary: None,
        };

        assert_eq!(
            apply_replicated_page(&manager, "u", vec![(7, frame)], Some(&leader)).unwrap(),
            1
        );

        let stored = follower.read().get_memory(&memory.id).unwrap();
        assert!(
            stored.experience.embeddings.is_none(),
            "the follower's primary slot holds only its own space"
        );
        assert_eq!(
            stored.experience.vector_in(&leader_space, None, None),
            Some(&vector)
        );
        assert_eq!(follower.read().embed_queue_len(), 1);

        let writer = manager.get_user_journaled_writer("u").unwrap();
        let last = writer.lock().log().iter().unwrap().last().unwrap().unwrap();
        let (_, journaled) = payload::decode_record(&last).unwrap();
        let IntentPayload::Remember { memory_bincode, .. } = journaled else {
            panic!("{journaled:?}");
        };
        let journaled = crate::memory::storage::decode_memory_snapshot(&memory_bincode).unwrap();
        assert!(journaled.experience.embeddings.is_none());
        assert_eq!(
            journaled.experience.vector_in(&leader_space, None, None),
            Some(&vector)
        );
    }

    /// A leader that declares no spaces is read as before spaces existed: an unrecorded vector
    /// as wide as the follower's is the follower's.
    #[test]
    fn a_leader_that_declares_no_spaces_is_read_as_before() {
        let (manager, _tmp) = crate::handlers::test_helpers::TestHarness::fresh_manager();
        let follower = manager.get_user_memory("u").unwrap();
        let own = follower.read().slot_spaces();
        let vector = unit(own.primary.dimension());
        let (memory, frame) = unstamped_snapshot(vector.clone());

        apply_replicated_page(&manager, "u", vec![(7, frame)], None).unwrap();

        let stored = follower.read().get_memory(&memory.id).unwrap();
        assert_eq!(stored.experience.embeddings.as_ref(), Some(&vector));
        assert_eq!(
            stored.experience.embedding_space.as_ref(),
            Some(&own.primary)
        );
    }

    #[test]
    fn a_page_lists_the_spaces_its_snapshots_hold() {
        let primary = SpaceId::new("leader-model", 4);
        let kept = SpaceId::new("older-model", 2);
        let (mut memory, _) = unstamped_snapshot(unit(4));
        memory.experience.keep_vector(kept.clone(), unit(2));
        let frame = IntentPayload::Update {
            user_id: "u".into(),
            memory_id: memory.id.0.to_string(),
            memory_bincode: bincode::serde::encode_to_vec(&memory, bincode::config::standard())
                .unwrap(),
            schema_version: Some(payload::CURRENT_PAYLOAD_SCHEMA_VERSION),
        };
        let records = vec![FeedRecord {
            lsn: 1,
            frame_b64: base64::engine::general_purpose::STANDARD
                .encode(payload::encode(&frame).unwrap()),
        }];
        let slot = SlotSpaces {
            primary: primary.clone(),
            secondary: None,
        };
        assert_eq!(page_spaces(&records, Some(&slot)), vec![primary, kept]);
        assert_eq!(
            page_spaces(&records, None),
            vec![SpaceId::new("older-model", 2), SpaceId::unrecorded(4)]
        );
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
        assert!(
            divergence(&caught_up, &page(100, 8)).is_some(),
            "log recreated"
        );
        assert!(
            divergence(&caught_up, &page(99, 7)).is_some(),
            "head at the applied lsn"
        );
        assert!(
            divergence(&caught_up, &page(40, 7)).is_some(),
            "rolled back"
        );
        assert!(
            divergence(&caught_up, &page(100, 0)).is_none(),
            "legacy log has no identity"
        );
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
        let saved = status
            .record_page(3, "u", &p, 9, 10, 4)
            .expect("current poll");
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

    fn record(lsn: u64) -> std::result::Result<crate::intent_log::IntentRecord, std::io::Error> {
        Ok(crate::intent_log::IntentRecord {
            lsn: crate::intent_log::Lsn(lsn),
            payload: vec![lsn as u8],
        })
    }

    fn lsns(page: &[FeedRecord]) -> Vec<u64> {
        page.iter().map(|r| r.lsn).collect()
    }

    #[test]
    fn the_feed_ends_below_the_head_it_captured() {
        let torn = || std::iter::once(Err(std::io::Error::other("truncated frame")));

        let appended_after_capture = (0..5).map(record).chain(torn());
        let (page, more) = page_below_head(appended_after_capture, 0, 4, 500).unwrap();
        assert_eq!(
            lsns(&page),
            vec![0, 1, 2, 3],
            "frame 4 came after the head was read"
        );
        assert!(!more);

        let (page, _) = page_below_head((0..4).map(record).chain(torn()), 2, 4, 500)
            .expect("a frame being flushed past the head is never read");
        assert_eq!(lsns(&page), vec![2, 3]);

        let (page, more) = page_below_head((0..4).map(record), 0, 4, 2).unwrap();
        assert_eq!(lsns(&page), vec![0, 1]);
        assert!(more);
        let (page, more) = page_below_head((0..2).map(record), 0, 2, 2).unwrap();
        assert_eq!(lsns(&page), vec![0, 1]);
        assert!(!more, "the page that reaches the head is the last");

        let (page, more) = page_below_head((0..4).map(record).chain(torn()), 4, 4, 500).unwrap();
        assert!(page.is_empty() && !more);
    }

    /// The floor is the lowest cursor asked for within the window; a lower one
    /// wins, hours older than the window drop out, and the floors survive a
    /// restart.
    #[test]
    fn feed_floors_keep_the_lowest_cursor_inside_the_window() {
        let base = tempfile::TempDir::new().unwrap();
        let day = Duration::from_secs(86_400);
        let window = 7 * day;
        let now = std::time::SystemTime::now();
        let mut floors = FeedFloors::load(base.path());
        assert_eq!(floors.floor("alice", window, now), None, "no follower seen");

        floors.note("alice", 40, window, now - 3 * day);
        floors.note("alice", 90, window, now - day);
        floors.note("alice", 70, window, now - day);
        floors.note("bob", 5, window, now);
        assert_eq!(floors.floor("alice", window, now), Some(40));
        assert_eq!(floors.floor("bob", window, now), Some(5));

        let reloaded = FeedFloors::load(base.path());
        assert_eq!(reloaded.floor("alice", window, now), Some(40));

        let later = now + 5 * day;
        assert_eq!(
            floors.floor("alice", window, later),
            Some(70),
            "the low hour aged out of the window"
        );
        floors.note("alice", 120, window, later);
        assert_eq!(floors.floor("alice", window, later), Some(70));
    }

    /// A follower whose cursor lies inside a range the leader erased (a forgotten
    /// memory's frames purged from a sealed segment) reads on from the next frame
    /// left: the page starts past the gap, as `feed` pages it. A follower that had
    /// applied the memory before its `Forget` was erased keeps it, which is why the
    /// purge window must outlast follower lag.
    #[test]
    fn a_cursor_inside_an_erased_range_reads_on_past_it() {
        use crate::intent_log::{payload, purge, IntentLog, Lsn};
        let snapshot = |id: &str| {
            payload::encode(&IntentPayload::Remember {
                user_id: "u".to_string(),
                memory_id: id.to_string(),
                memory_bincode: format!("{id}-content").into_bytes(),
                schema_version: None,
            })
            .unwrap()
        };
        let forget = |id: &str| {
            payload::encode(&IntentPayload::Forget {
                user_id: "u".to_string(),
                memory_id: id.to_string(),
                schema_version: None,
            })
            .unwrap()
        };
        let dir = tempfile::TempDir::new().unwrap();
        let mut log = IntentLog::open(dir.path().join("intent.log")).unwrap();
        log.set_segment_bytes(Some(120));
        for frame in [
            snapshot("a"),
            snapshot("b"),
            snapshot("a"),
            forget("a"),
            snapshot("c"),
            snapshot("b"),
        ] {
            log.append(&frame).unwrap();
            log.sync().unwrap();
        }
        let ten_days_ago =
            std::time::SystemTime::now() - std::time::Duration::from_secs(10 * 86_400);
        for segment in log.sealed() {
            std::fs::OpenOptions::new()
                .write(true)
                .open(&segment.path)
                .unwrap()
                .set_modified(ten_days_ago)
                .unwrap();
        }
        let report = purge::purge_forgotten(
            log.path(),
            None,
            std::time::Duration::from_secs(7 * 86_400),
            None,
            std::time::SystemTime::now(),
        )
        .unwrap();
        assert_eq!(report.frames_dropped, 3, "a's two snapshots and its Forget");
        let head = log.next_lsn().0;
        let page_from = |floor: u64, limit: usize| {
            page_below_head(log.iter_from(Lsn(floor)).unwrap(), floor, head, limit).unwrap()
        };

        let (page, more) = page_from(1, 500);
        assert_eq!(lsns(&page), vec![1, 4, 5]);
        assert!(!more);
        let (page, more) = page_from(2, 500);
        assert_eq!(
            lsns(&page),
            vec![4, 5],
            "a cursor on an erased LSN resumes at the next frame"
        );
        assert!(!more);
        let (page, more) = page_from(2, 1);
        assert_eq!(lsns(&page), vec![4]);
        assert!(more);
        let (page, more) = page_from(5, 1);
        assert_eq!(lsns(&page), vec![5]);
        assert!(!more);
    }

    fn frame(lsn: u64, id: &str, kind: &str) -> (u64, IntentPayload) {
        frame_for("u", lsn, id, kind)
    }

    fn frame_for(user: &str, lsn: u64, id: &str, kind: &str) -> (u64, IntentPayload) {
        let sv = Some(payload::CURRENT_PAYLOAD_SCHEMA_VERSION);
        let user_id = user.to_string();
        let p = match kind {
            "remember" => IntentPayload::Remember {
                user_id,
                memory_id: id.into(),
                memory_bincode: vec![lsn as u8],
                schema_version: sv,
            },
            "update" => IntentPayload::Update {
                user_id,
                memory_id: id.into(),
                memory_bincode: vec![lsn as u8],
                schema_version: sv,
            },
            "forget" => IntentPayload::Forget {
                user_id,
                memory_id: id.into(),
                schema_version: sv,
            },
            _ => IntentPayload::Anchor {
                user_id,
                memory_id: id.into(),
                importance: 0.9,
                schema_version: sv,
            },
        };
        (lsn, p)
    }

    #[test]
    fn lag_counts_the_leader_frames_not_yet_applied() {
        let mut p = UserProgress {
            leader_head_lsn: Some(41_000),
            applied_lsn: Some(16_499),
            ..Default::default()
        };
        p.refresh_lag();
        assert_eq!(p.lag, Some(24_500));
        p.applied_lsn = Some(40_999);
        p.refresh_lag();
        assert_eq!(
            p.lag,
            Some(0),
            "caught up: the leader's last LSN is head - 1"
        );
        p.applied_lsn = None;
        p.refresh_lag();
        assert_eq!(p.lag, Some(41_000), "nothing applied: the whole log");
        p.leader_head_lsn = None;
        p.refresh_lag();
        assert_eq!(p.lag, None);
    }

    #[tokio::test]
    async fn the_follower_returns_once_shutdown_is_cancelled() {
        let (manager, _tmp) = crate::handlers::test_helpers::TestHarness::fresh_manager();
        let shutdown = CancellationToken::new();
        let follower = tokio::spawn(run_follower(Arc::new(manager), shutdown.clone()));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!follower.is_finished(), "the follower runs until cancelled");
        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(1), follower)
            .await
            .expect("the follower returns within a second of the cancel, not after its interval")
            .unwrap();
    }

    #[test]
    fn a_release_that_keeps_importance_reaches_the_follower() {
        let (manager, _tmp) = crate::handlers::test_helpers::TestHarness::fresh_manager();
        let memory_sys = manager.get_user_memory("u").unwrap();
        let id = memory_sys
            .read()
            .remember(
                crate::memory::types::Experience {
                    content: "the deploy key rotates on the first of the month".into(),
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let memory = memory_sys.read().get_memory(&id).unwrap();
        memory.set_anchored(true);
        memory_sys.read().update_memory(&memory).unwrap();
        let importance = memory.importance();
        let anchored = || memory_sys.read().get_memory(&id).unwrap().is_anchored();
        let sv = Some(payload::CURRENT_PAYLOAD_SCHEMA_VERSION);

        let legacy = IntentPayload::Anchor {
            user_id: "u".into(),
            memory_id: id.0.to_string(),
            importance,
            schema_version: sv,
        };
        apply_non_snapshot(&manager, "u", &legacy).unwrap();
        assert!(
            anchored(),
            "a legacy frame cannot say the anchor was released"
        );

        let release = IntentPayload::SetAnchored {
            user_id: "u".into(),
            memory_id: id.0.to_string(),
            anchored: false,
            importance,
            schema_version: sv,
        };
        apply_non_snapshot(&manager, "u", &release).unwrap();
        assert!(!anchored());
        assert_eq!(
            memory_sys.read().get_memory(&id).unwrap().importance(),
            importance
        );
    }

    #[test]
    fn a_page_lands_in_the_store_and_the_local_log_in_one_batch() {
        let (manager, _tmp) = crate::handlers::test_helpers::TestHarness::fresh_manager();
        let sv = Some(payload::CURRENT_PAYLOAD_SCHEMA_VERSION);
        let snapshot = |content: &str| {
            let memory = crate::memory::types::Memory::new(
                MemoryId(uuid::Uuid::new_v4()),
                crate::memory::types::Experience {
                    content: content.to_string(),
                    ..Default::default()
                },
                0.5,
                None,
                None,
                None,
                None,
            );
            let bytes =
                bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
            (memory.id, bytes)
        };
        let (first, first_bytes) = snapshot("the harbour crane is serviced in spring");
        let (second, second_bytes) = snapshot("the tide gauge reads high on Thursdays");
        let frames = vec![
            (
                10,
                IntentPayload::Remember {
                    user_id: "u".into(),
                    memory_id: first.0.to_string(),
                    memory_bincode: first_bytes,
                    schema_version: sv,
                },
            ),
            (
                11,
                IntentPayload::Remember {
                    user_id: "u".into(),
                    memory_id: second.0.to_string(),
                    memory_bincode: second_bytes,
                    schema_version: sv,
                },
            ),
            (
                12,
                IntentPayload::SetAnchored {
                    user_id: "u".into(),
                    memory_id: first.0.to_string(),
                    anchored: true,
                    importance: 0.9,
                    schema_version: sv,
                },
            ),
        ];

        assert_eq!(
            apply_replicated_page(&manager, "u", frames, None).unwrap(),
            3
        );

        let memory_sys = manager.get_user_memory("u").unwrap();
        assert!(memory_sys.read().get_memory(&first).unwrap().is_anchored());
        assert!(memory_sys.read().get_memory(&second).is_ok());
        let writer = manager.get_user_journaled_writer("u").unwrap();
        let mut journaled: Vec<String> = writer
            .lock()
            .log()
            .iter()
            .unwrap()
            .map(|record| {
                let (_, payload) = payload::decode_record(&record.unwrap()).unwrap();
                payload.memory_id().to_string()
            })
            .collect();
        journaled.sort();
        let mut expected = vec![
            first.0.to_string(),
            first.0.to_string(),
            second.0.to_string(),
        ];
        expected.sort();
        assert_eq!(journaled, expected, "the page's frames, once each");
    }

    /// An update, a bulk forget and a batch remember on the leader reach a follower
    /// through the leader's log, with their ids: the paths that did not journal before.
    #[tokio::test]
    async fn an_update_a_bulk_forget_and_a_batch_remember_reach_the_follower() {
        use crate::handlers::test_helpers::{post_json, put_json, send, TestHarness};
        use axum::http::StatusCode;

        let leader = TestHarness::new();
        let user = "replica-user";
        let (status, body) = send(
            leader.router(),
            post_json(
                "/api/remember/batch",
                &serde_json::json!({
                    "user_id": user,
                    "memories": [
                        {"content": "the dredger works the channel at night", "tags": ["keep"]},
                        {"content": "the old pier is closed to walkers", "tags": ["sweep"]},
                        {"content": "the lighthouse lamp was replaced in May", "tags": ["sweep"]},
                    ],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let ids: Vec<String> = body["memory_ids"]
            .as_array()
            .expect("batch remember returns its ids")
            .iter()
            .map(|id| id.as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids.len(), 3, "{body}");

        let revised = "the dredger works the channel at dawn";
        let (status, body) = send(
            leader.router(),
            put_json(
                &format!("/api/memory/{}", ids[0]),
                &serde_json::json!({ "user_id": user, "content": revised }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let (status, body) = send(
            leader.router(),
            post_json(
                "/api/memories/bulk",
                &serde_json::json!({ "user_id": user, "tags": ["sweep"] }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");

        let frames: Vec<(u64, IntentPayload)> = {
            let writer = leader.state().get_user_journaled_writer(user).unwrap();
            let writer = writer.lock();
            writer
                .log()
                .iter()
                .unwrap()
                .map(|record| {
                    let (lsn, payload) = payload::decode_record(&record.unwrap()).unwrap();
                    (lsn.0, payload)
                })
                .collect()
        };
        let (follower, _tmp) = TestHarness::fresh_manager();
        let follower = Arc::new(follower);
        let applied = {
            let follower = Arc::clone(&follower);
            tokio::task::spawn_blocking(move || {
                apply_replicated_page(&follower, user, frames, None)
            })
            .await
            .unwrap()
            .unwrap()
        };
        assert!(applied > 0);

        let replica = follower.get_user_memory(user).unwrap();
        let replica = replica.read();
        let id = |text: &str| MemoryId(uuid::Uuid::parse_str(text).unwrap());
        assert_eq!(
            replica.get_memory(&id(&ids[0])).unwrap().experience.content,
            revised
        );
        for swept in &ids[1..] {
            assert!(
                replica.get_memory(&id(swept)).is_err(),
                "{swept} survived its bulk forget on the follower"
            );
        }
    }

    #[test]
    fn a_page_naming_another_tenant_is_refused() {
        let ok = vec![frame(0, "a", "remember"), frame(1, "b", "forget")];
        assert!(check_tenant("u", &ok).is_ok());
        let mixed = vec![
            frame(0, "a", "remember"),
            frame_for("other", 1, "b", "forget"),
        ];
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
        let page = vec![
            frame(0, "a", "remember"),
            frame(1, "a", "forget"),
            frame(2, "a", "remember"),
        ];
        let out: Vec<u64> = coalesce(page).into_iter().map(|(l, _)| l).collect();
        assert_eq!(out, vec![2]);
    }

    #[test]
    fn the_leader_lists_only_tenants_it_can_feed() {
        let (manager, _tmp) = crate::handlers::test_helpers::TestHarness::fresh_manager();
        manager.get_user_journaled_writer("journaled").unwrap();
        std::fs::create_dir_all(manager.user_dir("copied-aside").unwrap()).unwrap();
        let listed: Vec<String> = user_logs(&manager).into_iter().map(|u| u.user_id).collect();
        assert!(listed.contains(&"journaled".to_string()));
        assert!(
            !listed.contains(&"copied-aside".to_string()),
            "a directory with no intent log is not a tenant the feed can serve"
        );
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
        let back = payload::decode(
            &base64::engine::general_purpose::STANDARD
                .decode(b64)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(back.memory_id(), "m");
    }

    fn tls_fixture(name: &str) -> String {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tls")
            .join(name)
            .display()
            .to_string()
    }

    fn tls_from(vars: &[(&str, &str)]) -> Result<LeaderTls> {
        let vars: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        LeaderTls::from_lookup(|k| vars.get(k).cloned())
    }

    fn tls_refusal(vars: &[(&str, &str)]) -> String {
        let err = tls_from(vars).expect_err("the TLS variables are refused");
        format!("{err:#}")
    }

    #[test]
    fn no_tls_variables_give_a_plain_client() {
        let tls = tls_from(&[]).unwrap();
        assert!(tls.roots.is_empty());
        assert!(tls.identity.is_none());
        tls.client(PROBE_TIMEOUT).unwrap();
    }

    #[test]
    fn a_client_identity_needs_both_halves() {
        let cert = tls_fixture("test-ca-bundle.pem");
        let only_cert = tls_refusal(&[(ENV_CLIENT_CERT, &cert)]);
        assert!(
            only_cert.contains(&format!(
                "{ENV_CLIENT_CERT} is set but {ENV_CLIENT_KEY} is not"
            )),
            "{only_cert}"
        );
        let only_key = tls_refusal(&[(ENV_CLIENT_CERT, "  "), (ENV_CLIENT_KEY, &cert)]);
        assert!(
            only_key.contains(&format!(
                "{ENV_CLIENT_KEY} is set but {ENV_CLIENT_CERT} is not"
            )),
            "{only_key}"
        );
    }

    #[test]
    fn an_unreadable_ca_bundle_names_the_variable_and_the_path() {
        let missing = tls_fixture("no-such-bundle.pem");
        let err = tls_refusal(&[(ENV_CA_CERT, &missing)]);
        assert!(err.contains(&format!("{ENV_CA_CERT}={missing}")), "{err}");
    }

    #[test]
    fn a_bad_pem_names_the_variable_and_the_path() {
        let corrupt = tls_fixture("corrupt-cert.pem");
        let err = tls_refusal(&[(ENV_CA_CERT, &corrupt)]);
        assert!(
            err.contains(&format!(
                "{ENV_CA_CERT}={corrupt}: not a PEM certificate bundle"
            )),
            "{err}"
        );

        let not_pem = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("Cargo.toml")
            .display()
            .to_string();
        let err = tls_refusal(&[(ENV_CA_CERT, &not_pem)]);
        assert!(
            err.contains(&format!("{ENV_CA_CERT}={not_pem}: no PEM certificate")),
            "{err}"
        );
        let err = tls_refusal(&[(ENV_CLIENT_CERT, &not_pem), (ENV_CLIENT_KEY, &not_pem)]);
        assert!(
            err.contains(&format!("{ENV_CLIENT_CERT}={not_pem}: no PEM certificate")),
            "{err}"
        );
    }

    #[test]
    fn a_ca_bundle_adds_every_certificate_in_it() {
        let bundle = tls_fixture("test-ca-bundle.pem");
        let tls = tls_from(&[(ENV_CA_CERT, &bundle)]).unwrap();
        assert_eq!(tls.roots.len(), 2);
        assert_eq!(tls.files, vec![format!("{ENV_CA_CERT}={bundle}")]);
        assert!(tls.identity.is_none());
        tls.client(PROBE_TIMEOUT).unwrap();
    }

    #[test]
    fn a_client_identity_without_a_key_names_both_files() {
        let cert = tls_fixture("test-ca-bundle.pem");
        let err = tls_refusal(&[(ENV_CLIENT_CERT, &cert), (ENV_CLIENT_KEY, &cert)]);
        assert!(
            err.contains(&format!(
                "{ENV_CLIENT_CERT}={cert} with {ENV_CLIENT_KEY}={cert}"
            )),
            "{err}"
        );

        let missing = tls_fixture("no-such-key.pem");
        let err = tls_refusal(&[(ENV_CLIENT_CERT, &cert), (ENV_CLIENT_KEY, &missing)]);
        assert!(
            err.contains(&format!("{ENV_CLIENT_KEY}={missing}")),
            "{err}"
        );
    }
}
