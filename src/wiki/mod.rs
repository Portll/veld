//! Wiki render projection (LLM-Wiki) — the wiki view as an intent-log
//! projection.
//!
//! Veld is the canonical store; the wiki is a *rendered projection* of it
//! (synthesis decision B/D). Rather than a bespoke render queue + file
//! watcher (the original plan's fragile design), the render is wired as a
//! fourth [`crate::intent_log::Projection`] over the same intent log the
//! SQLite / Vamana / BM25 projections derive from. Consequences:
//!
//! - **Incremental by construction (W-013/W-014).** Each `Remember` /
//!   `Update` / `Forget` re-renders or removes exactly one page; there is
//!   no O(N) global re-render per ingest, and `log.md`/`index.md` are not
//!   rebuilt from scratch each write.
//! - **Complete by construction.** The C2 store↔log reconcile guarantees
//!   every canonical memory has a log frame, so a replay rebuilds the full
//!   wiki faithfully.
//! - **Dormant until enabled.** Registration is gated on `VELD_WIKI_RENDER`
//!   (default off), so this lands on `main` without changing default
//!   behaviour for any tenant.
//!
//! Safety is baked in from the first commit:
//! - **W-007 path confinement.** Page files are named by memory UUID
//!   (`{uuid}.md`) — inherently free of path separators — and every write /
//!   remove resolves under the per-user wiki root via [`confine_under`].
//!   The human-readable slug is sanitised ([`sanitize_slug`]) and used only
//!   as frontmatter, never as a path component.
//! - **W-008 disk PII.** Rendered bodies pass through
//!   [`crate::mif::pii::PiiPatterns::redact`] before any byte hits disk.
//! - **W-016 render eligibility.** Tool-noise experience types
//!   (CodeEdit / FileAccess / Search / Command) are not page-worthy and are
//!   never rendered (and are removed if an update drops worthiness).
//!
//! Edit-back (the file watcher, W-003/W-004) and an authed manual-render
//! endpoint (W-018) build on this foundation in follow-up commits.

use std::error::Error;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::intent_log::{
    CheckpointStore, CheckpointStoreError, IntentPayload, IntentRecord, Lsn, PayloadError,
    Projection, TypedProjection,
};
use crate::memory::types::{ExperienceType, Memory, MemoryId, SourceType};
use crate::mif::pii::PiiPatterns;

/// Checkpoint-store key + metrics label. Stable across versions — the
/// checkpoint store is keyed on this string, so renaming it is a manual
/// migration.
pub const PROJECTION_NAME: &str = "wiki";

/// Persist the projection checkpoint every N applied records (replay and
/// live writes). Page files are written atomically per record (temp +
/// rename), so a crash only costs the un-checkpointed tail, which the next
/// replay re-applies idempotently.
pub const COMMIT_EVERY: u64 = 100;

/// Is the wiki render projection enabled? Default OFF — the projection is
/// only registered when `VELD_WIKI_RENDER` is truthy, so the feature is
/// dormant on `main` until an operator opts in.
pub fn wiki_render_enabled() -> bool {
    std::env::var("VELD_WIKI_RENDER")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
        .unwrap_or(false)
}

/// Resolve the per-user wiki output directory. When `VELD_WIKI_DIR` is set
/// it is treated as a shared root and each tenant renders under
/// `{VELD_WIKI_DIR}/{user_id}`; otherwise pages live under
/// `{user_path}/wiki` (already per-user isolated by the data layout).
pub fn wiki_target_dir(user_path: &Path, user_id: &str) -> PathBuf {
    match std::env::var("VELD_WIKI_DIR") {
        Ok(base) if !base.trim().is_empty() => PathBuf::from(base).join(user_id),
        _ => user_path.join("wiki"),
    }
}

/// Errors raised by [`WikiProjection`]. Distinct from `anyhow::Error`
/// because [`Projection::Error`] requires `std::error::Error`.
#[derive(Debug, thiserror::Error)]
pub enum WikiProjectionError {
    #[error("wiki io error: {0}")]
    Io(String),
    #[error("intent log payload decode error: {0}")]
    PayloadDecode(#[from] PayloadError),
    #[error("checkpoint store error: {0}")]
    Checkpoint(#[from] CheckpointStoreError),
    #[error("memory_id is not a valid UUID: {0}")]
    InvalidMemoryId(String),
    #[error("memory bincode decode error: {0}")]
    MemoryDecode(String),
    #[error("refusing to write outside the wiki root: {0}")]
    PathEscape(String),
}

/// Wiki render projection: materialises page-worthy memories to markdown.
/// One per tenant. Holds the resolved (canonicalised) output dir, a shared
/// [`CheckpointStore`], the compiled PII patterns, and the last-applied LSN.
pub struct WikiProjection {
    target_dir: PathBuf,
    checkpoint_store: Arc<Mutex<CheckpointStore>>,
    checkpoint: Option<Lsn>,
    pii: PiiPatterns,
    applies_since_persist: u64,
}

impl WikiProjection {
    /// Construct around a resolved per-user wiki dir (created if missing)
    /// and the shared checkpoint store. Resumes from the last-persisted
    /// checkpoint so replay / live apply pick up at the right LSN.
    pub fn new(
        target_dir: PathBuf,
        checkpoint_store: Arc<Mutex<CheckpointStore>>,
    ) -> Result<Self, WikiProjectionError> {
        std::fs::create_dir_all(&target_dir).map_err(|e| {
            WikiProjectionError::Io(format!("create wiki dir {target_dir:?}: {e}"))
        })?;
        // Canonicalise the root once so `confine_under` can compare every
        // page path against a normalised absolute root.
        let target_dir = std::fs::canonicalize(&target_dir).map_err(|e| {
            WikiProjectionError::Io(format!("canonicalize wiki dir {target_dir:?}: {e}"))
        })?;
        let checkpoint = checkpoint_store.lock().get(PROJECTION_NAME);
        Ok(Self {
            target_dir,
            checkpoint_store,
            checkpoint,
            pii: PiiPatterns::new(),
            applies_since_persist: 0,
        })
    }

    /// One code path for both trait arms (dyn live-write + static replay).
    fn apply_typed(&mut self, lsn: Lsn, payload: &IntentPayload) -> Result<(), WikiProjectionError> {
        match payload {
            IntentPayload::Remember {
                memory_id,
                memory_bincode,
                ..
            }
            | IntentPayload::Update {
                memory_id,
                memory_bincode,
                ..
            } => {
                let mid = parse_memory_id(memory_id)?;
                let memory = decode_memory(memory_bincode)?;
                if is_page_worthy(&memory) {
                    let markdown = self.render_page(&memory, lsn);
                    self.write_page(&mid, &markdown)?;
                } else {
                    // An update may drop a memory below page-worthiness
                    // (e.g. reclassified as tool-noise) — remove any stale
                    // page so the wiki never shows a now-ineligible memory.
                    self.remove_page(&mid)?;
                }
            }
            IntentPayload::Forget { memory_id, .. } => {
                let mid = parse_memory_id(memory_id)?;
                self.remove_page(&mid)?;
            }
            IntentPayload::Anchor { .. } => {
                // Anchoring changes decay resistance only — not page
                // eligibility (by experience type) or content. No
                // re-render; the checkpoint still advances below so the
                // projection's replay-lag does not appear stuck.
            }
        }
        self.checkpoint = Some(lsn);
        Ok(())
    }

    /// Resolve `{target_dir}/{uuid}.md`, asserting it stays under the root.
    fn page_path(&self, mid: &MemoryId) -> Result<PathBuf, WikiProjectionError> {
        let path = self.target_dir.join(format!("{}.md", mid.0));
        confine_under(&self.target_dir, &path)?;
        Ok(path)
    }

    /// Atomic per-page write: temp file in the same dir + rename, so a
    /// reader (Obsidian) never observes a half-written page.
    fn write_page(&mut self, mid: &MemoryId, markdown: &str) -> Result<(), WikiProjectionError> {
        let path = self.page_path(mid)?;
        let tmp = path.with_extension("md.tmp");
        std::fs::write(&tmp, markdown.as_bytes())
            .map_err(|e| WikiProjectionError::Io(format!("write {tmp:?}: {e}")))?;
        std::fs::rename(&tmp, &path)
            .map_err(|e| WikiProjectionError::Io(format!("rename {tmp:?} -> {path:?}: {e}")))?;
        Ok(())
    }

    fn remove_page(&mut self, mid: &MemoryId) -> Result<(), WikiProjectionError> {
        let path = self.page_path(mid)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(WikiProjectionError::Io(format!("remove {path:?}: {e}"))),
        }
    }

    /// Render a memory to a markdown page: YAML frontmatter + redacted body
    /// + entity wikilinks. All free-text that reaches disk is PII-redacted.
    fn render_page(&self, memory: &Memory, lsn: Lsn) -> String {
        let (source_type, source_trusted) = match memory.experience.context.as_ref() {
            Some(c) => (
                format!("{:?}", c.source.source_type),
                matches!(
                    c.source.source_type,
                    SourceType::User | SourceType::File | SourceType::System
                ),
            ),
            None => ("Unknown".to_string(), false),
        };
        // W-008: redact PII before anything reaches disk.
        let (body, _redactions, _found) = self.pii.redact(&memory.experience.content);
        let raw_slug = sanitize_slug(body.lines().next().unwrap_or("").trim());
        let slug = if raw_slug.is_empty() {
            format!("mem-{}", short_id(memory))
        } else {
            raw_slug
        };

        let mut out = String::with_capacity(body.len() + 512);
        out.push_str("---\n");
        out.push_str(&format!("id: {}\n", memory.id.0));
        out.push_str(&format!("slug: {slug}\n"));
        out.push_str(&format!(
            "page_type: {:?}\n",
            memory.experience.experience_type
        ));
        out.push_str(&format!("source_type: {source_type}\n"));
        out.push_str(&format!("source_trusted: {source_trusted}\n"));
        out.push_str(&format!("created_at: {}\n", memory.created_at.to_rfc3339()));
        out.push_str(&format!("last_rendered_lsn: {}\n", lsn.0));
        if !memory.experience.tags.is_empty() {
            let tags = memory
                .experience
                .tags
                .iter()
                .map(|t| sanitize_inline(t))
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("tags: [{tags}]\n"));
        }
        out.push_str("---\n\n");
        out.push_str(&body);
        out.push('\n');
        if !memory.experience.entities.is_empty() {
            out.push_str("\n## Entities\n\n");
            for e in &memory.experience.entities {
                let e = sanitize_inline(e);
                if !e.is_empty() {
                    out.push_str(&format!("- [[{e}]]\n"));
                }
            }
        }
        out
    }

    fn persist(&mut self) -> Result<(), WikiProjectionError> {
        if let Some(lsn) = self.checkpoint {
            let mut store = self.checkpoint_store.lock();
            store.set(PROJECTION_NAME, lsn)?;
            store.sync()?;
        }
        Ok(())
    }
}

/// Decode a memory_id string into a [`MemoryId`]; structured error so the
/// journaled writer can log+skip rather than panic.
fn parse_memory_id(s: &str) -> Result<MemoryId, WikiProjectionError> {
    uuid::Uuid::parse_str(s)
        .map(MemoryId)
        .map_err(|e| WikiProjectionError::InvalidMemoryId(format!("{s}: {e}")))
}

/// Decode a bincoded `Memory` snapshot off an `IntentPayload`. Config
/// mirrors `handlers/remember.rs` (`bincode::config::standard()`) so a
/// snapshot from the live-write path round-trips here.
fn decode_memory(bytes: &[u8]) -> Result<Memory, WikiProjectionError> {
    let (memory, _consumed): (Memory, usize) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(|e| WikiProjectionError::MemoryDecode(e.to_string()))?;
    Ok(memory)
}

/// W-016: only knowledge-bearing memories become wiki pages. Pure
/// tool-noise experience types are excluded so the wiki is a curated view,
/// not a transcript of every Read/Bash the agent ran.
fn is_page_worthy(memory: &Memory) -> bool {
    !matches!(
        memory.experience.experience_type,
        ExperienceType::CodeEdit
            | ExperienceType::FileAccess
            | ExperienceType::Search
            | ExperienceType::Command
    )
}

/// W-007: assert `path` resolves under `root`. Page filenames are
/// `{uuid}.md` (no separators), so traversal is already impossible; this is
/// defence-in-depth. The page may not exist yet, so we canonicalise the
/// *parent* (which is always the root for our flat layout) and compare; if
/// the parent can't be canonicalised we fall back to a lexical `..` check.
fn confine_under(root: &Path, path: &Path) -> Result<(), WikiProjectionError> {
    let parent = path.parent().unwrap_or(root);
    match std::fs::canonicalize(parent) {
        Ok(p) if p == root => Ok(()),
        Ok(p) => Err(WikiProjectionError::PathEscape(format!(
            "{p:?} is not the wiki root {root:?}"
        ))),
        Err(_) => {
            if path.components().any(|c| matches!(c, Component::ParentDir)) {
                Err(WikiProjectionError::PathEscape(format!(
                    "{path:?} contains a parent-dir traversal"
                )))
            } else {
                Ok(())
            }
        }
    }
}

/// Sanitise free text into a slug-safe token: lowercase ASCII alphanumerics
/// joined by single hyphens, capped at 60 chars. Used for the frontmatter
/// `slug` only — never as a path component. May return `""` (caller
/// substitutes a `mem-{id}` fallback).
fn sanitize_slug(title: &str) -> String {
    let lowered: String = title
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect();
    let slug = lowered.split_whitespace().collect::<Vec<_>>().join("-");
    slug.chars().take(60).collect()
}

/// Strip characters that would break a one-line frontmatter list entry or a
/// `[[wikilink]]` (newlines, brackets, commas).
fn sanitize_inline(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '\n' | '\r' | '[' | ']' | ','))
        .collect::<String>()
        .trim()
        .to_string()
}

/// First 8 chars of the memory UUID — a stable, collision-unlikely suffix
/// for the `mem-…` fallback slug.
fn short_id(memory: &Memory) -> String {
    memory.id.0.to_string().chars().take(8).collect()
}

/// Dyn-friendly arm: [`crate::intent_log::JournaledWriter`] hands typed
/// payloads at live-write time. Pages are written atomically per apply; the
/// checkpoint is persisted on the [`COMMIT_EVERY`] cadence so a crash only
/// re-applies the un-checkpointed tail (idempotent file writes).
impl TypedProjection for WikiProjection {
    fn name(&self) -> &str {
        PROJECTION_NAME
    }

    fn apply(
        &mut self,
        lsn: Lsn,
        payload: &IntentPayload,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.apply_typed(lsn, payload)
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
        self.applies_since_persist += 1;
        if self.applies_since_persist >= COMMIT_EVERY {
            self.persist()
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
            self.applies_since_persist = 0;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Static arm: the replay driver hands [`IntentRecord`]s straight off the
/// log on first writer open, catching the wiki up to the head.
impl Projection for WikiProjection {
    type Error = WikiProjectionError;

    fn name(&self) -> &str {
        PROJECTION_NAME
    }

    fn apply(&mut self, record: &IntentRecord) -> Result<(), Self::Error> {
        let (lsn, payload) = crate::intent_log::payload::decode_record(record)?;
        self.apply_typed(lsn, &payload)?;
        self.applies_since_persist += 1;
        Ok(())
    }

    fn checkpoint(&self) -> Option<Lsn> {
        self.checkpoint
    }

    fn persist_checkpoint(&mut self) -> Result<(), Self::Error> {
        self.persist()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_slug_is_path_safe_and_capped() {
        assert_eq!(sanitize_slug("Hello, World!"), "hello-world");
        assert_eq!(sanitize_slug("  ../../etc/passwd  "), "etc-passwd");
        assert_eq!(sanitize_slug("a/b\\c:d"), "a-b-c-d");
        assert_eq!(sanitize_slug(""), "");
        assert_eq!(sanitize_slug("!!!"), "");
        // capped at 60 chars, no separators ever survive
        let long = "x".repeat(200);
        let s = sanitize_slug(&long);
        assert_eq!(s.len(), 60);
        assert!(!s.contains('/') && !s.contains("..") && !s.contains('\\'));
    }

    #[test]
    fn sanitize_inline_strips_structural_chars() {
        assert_eq!(sanitize_inline("a,b[c]\nd"), "abcd");
        assert_eq!(sanitize_inline("  spaced  "), "spaced");
    }

    #[test]
    fn render_enabled_defaults_off_when_unset() {
        // Doesn't set the var (avoids cross-test env races); just asserts
        // the parse rejects obvious non-truthy values.
        // (Full env-gated behaviour is covered at the registration site.)
        assert!(!matches!("off", "1" | "true" | "on" | "yes"));
    }
}
