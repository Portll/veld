//! Per-tag provenance for a stored memory.
//!
//! `Experience::tags` is one flat list mixing what the writer chose with what veld derived (intent,
//! ingest hints, NER, YAKE, cold-start heuristics), and nothing on the record says which is which.
//! This ledger records the source of every tag that survived redaction and the cap, keyed by
//! memory id in the user's own KV keyspace.
//!
//! It lives beside the record rather than on it because `Memory` is bincode-encoded, which is
//! positional: a new field on any type inside a `Memory` makes every record written before it
//! undecodable, whatever `serde(default)` says (see `tests/persisted_layout.rs`).
//!
//! A memory with no row is UNDETERMINED. Rows are never back-filled by guessing from a tag's shape.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::memory::types::MemoryId;
use crate::redaction::RedactionReport;
use crate::storage::KeyValueStore;

pub(crate) const TAG_SOURCE_PREFIX: &str = "tag_source:";
pub const TAG_SOURCES_SCHEMA: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TagSource {
    Caller,
    Intent,
    Ingest,
    Ner,
    Yake,
    ColdStart,
}

impl TagSource {
    pub fn is_authored(self) -> bool {
        matches!(self, TagSource::Caller | TagSource::Intent)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TagSource::Caller => "caller",
            TagSource::Intent => "intent",
            TagSource::Ingest => "ingest",
            TagSource::Ner => "ner",
            TagSource::Yake => "yake",
            TagSource::ColdStart => "cold-start",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourcedTag {
    pub tag: String,
    pub source: TagSource,
    /// The value was credential-shaped and `tag` holds its class label instead.
    #[serde(default)]
    pub redacted: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactedCounts {
    pub key: usize,
    pub private_key: usize,
    pub token: usize,
    pub login: usize,
    pub secret: usize,
    pub path: usize,
}

impl From<&RedactionReport> for RedactedCounts {
    fn from(r: &RedactionReport) -> Self {
        Self {
            key: r.key,
            private_key: r.private_key,
            token: r.token,
            login: r.login,
            secret: r.secret,
            path: r.path,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TagSources {
    pub v: u32,
    pub tags: Vec<SourcedTag>,
    #[serde(default)]
    pub redacted: RedactedCounts,
}

impl TagSources {
    /// Label the tags that were finally stored.
    ///
    /// `sources`, `before` and `after` are aligned index for index: the assembled list before
    /// redaction, and the same list after `redact_entities` (which rewrites in place and keeps
    /// order). `kept` is what the cap returned — a subset of `after`, possibly reordered — so the
    /// join is by value, and the first occurrence wins, which is the authored one whenever an
    /// authored and a derived tag collide.
    pub fn label(
        sources: &[TagSource],
        before: &[String],
        after: &[String],
        kept: &[String],
        report: &RedactionReport,
    ) -> Self {
        debug_assert_eq!(sources.len(), before.len());
        debug_assert_eq!(before.len(), after.len());
        let mut by_tag: HashMap<&str, (TagSource, bool)> = HashMap::with_capacity(after.len());
        for ((source, b), a) in sources.iter().zip(before).zip(after) {
            by_tag.entry(a.as_str()).or_insert((*source, a != b));
        }
        let tags = kept
            .iter()
            .filter_map(|t| {
                let found = by_tag.get(t.as_str()).copied();
                debug_assert!(found.is_some(), "cap returned a tag it was not given");
                found.map(|(source, redacted)| SourcedTag {
                    tag: t.clone(),
                    source,
                    redacted,
                })
            })
            .collect();
        Self {
            v: TAG_SOURCES_SCHEMA,
            tags,
            redacted: RedactedCounts::from(report),
        }
    }

    /// The row for what was actually stored. `stored` is the record's tag list after the write;
    /// when the write merged into an existing record it is a union, so labels come from this write
    /// first and the existing row second. If any stored tag has a source neither knows, there is no
    /// row: a partially labelled record would read as fully determined.
    pub fn reconcile(
        incoming: TagSources,
        existing: Option<TagSources>,
        stored: &[String],
    ) -> Option<Self> {
        if incoming
            .tags
            .iter()
            .map(|t| t.tag.as_str())
            .eq(stored.iter().map(String::as_str))
        {
            return Some(incoming);
        }
        let mut redacted = incoming.redacted.clone();
        let mut by_tag: HashMap<String, SourcedTag> = HashMap::new();
        for t in incoming.tags {
            by_tag.entry(t.tag.clone()).or_insert(t);
        }
        if let Some(existing) = existing {
            for t in existing.tags {
                by_tag.entry(t.tag.clone()).or_insert(t);
            }
            redacted.key += existing.redacted.key;
            redacted.private_key += existing.redacted.private_key;
            redacted.token += existing.redacted.token;
            redacted.login += existing.redacted.login;
            redacted.secret += existing.redacted.secret;
            redacted.path += existing.redacted.path;
        }
        let tags = stored
            .iter()
            .map(|t| by_tag.get(t).cloned())
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            v: TAG_SOURCES_SCHEMA,
            tags,
            redacted,
        })
    }

    /// Every tag the caller sent, nothing derived — the write paths that run no extraction.
    pub fn all_caller(tags: &[String], before: &[String], report: &RedactionReport) -> Self {
        let sources = vec![TagSource::Caller; tags.len()];
        Self::label(&sources, before, tags, tags, report)
    }
}

/// Assembles a tag list and the source of each entry in step, so the two cannot drift apart.
#[derive(Debug, Default)]
pub struct TagAssembly {
    pub tags: Vec<String>,
    pub sources: Vec<TagSource>,
}

impl TagAssembly {
    pub fn push(&mut self, tag: String, source: TagSource) {
        self.tags.push(tag);
        self.sources.push(source);
    }

    pub fn extend(&mut self, tags: impl IntoIterator<Item = String>, source: TagSource) {
        for t in tags {
            self.push(t, source);
        }
    }

    pub fn len(&self) -> usize {
        self.tags.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }
}

pub struct TagSourceStore {
    kv: Arc<dyn KeyValueStore>,
}

impl TagSourceStore {
    pub fn with_kv(kv: Arc<dyn KeyValueStore>) -> Self {
        Self { kv }
    }

    fn key(id: &MemoryId) -> Vec<u8> {
        format!("{TAG_SOURCE_PREFIX}{}", id.0).into_bytes()
    }

    /// Replaces any existing row: an upsert's tags are the new truth, not an addition.
    pub fn put(&self, id: &MemoryId, sources: &TagSources) -> Result<()> {
        let bytes = serde_json::to_vec(sources).context("encode tag sources")?;
        self.kv.put_bytes(&Self::key(id), &bytes)
    }

    pub fn get(&self, id: &MemoryId) -> Result<Option<TagSources>> {
        match self.kv.get_bytes(&Self::key(id))? {
            None => Ok(None),
            Some(bytes) => serde_json::from_slice(&bytes)
                .with_context(|| format!("decode tag sources for {}", id.0))
                .map(Some),
        }
    }

    pub fn delete(&self, id: &MemoryId) -> Result<()> {
        self.kv.delete_bytes(&Self::key(id))
    }

    /// Every row. A scan error or an undecodable row is an error, never a shorter map: a reader
    /// that dropped rows would report those memories as undetermined when they are not.
    pub fn all(&self) -> Result<HashMap<MemoryId, TagSources>> {
        let rows = self
            .kv
            .scan_prefix(TAG_SOURCE_PREFIX.as_bytes(), None)
            .context("scan tag sources")?;
        let mut out = HashMap::with_capacity(rows.len());
        for (key, value) in rows {
            let key = String::from_utf8(key).context("tag source key is not utf-8")?;
            let id_str = key
                .strip_prefix(TAG_SOURCE_PREFIX)
                .context("tag source key without prefix")?;
            let id = MemoryId(
                uuid::Uuid::parse_str(id_str)
                    .with_context(|| format!("tag source key {key} is not a memory id"))?,
            );
            let sources: TagSources = serde_json::from_slice(&value)
                .with_context(|| format!("decode tag sources for {id_str}"))?;
            out.insert(id, sources);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::test_support::key_value_stores;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn label_joins_by_value_after_the_cap_reorders() {
        let sources = [TagSource::Caller, TagSource::Ner, TagSource::Yake];
        let before = s(&["project:veld", "Alice", "budget"]);
        let after = before.clone();
        let kept = s(&["budget", "project:veld"]);
        let got = TagSources::label(
            &sources,
            &before,
            &after,
            &kept,
            &RedactionReport::default(),
        );
        assert_eq!(got.tags.len(), 2);
        assert_eq!(got.tags[0].tag, "budget");
        assert_eq!(got.tags[0].source, TagSource::Yake);
        assert_eq!(got.tags[1].source, TagSource::Caller);
        assert!(got.tags.iter().all(|t| !t.redacted));
    }

    #[test]
    fn a_redacted_tag_keeps_its_source_and_is_flagged() {
        let mut after = s(&["ghp_abcdefghijklmnopqrstuvwxyz0123", "rust"]);
        let before = after.clone();
        let report = crate::redaction::redact_entities(&mut after);
        let sources = [TagSource::Caller, TagSource::Yake];
        let got = TagSources::label(&sources, &before, &after, &after, &report);
        assert!(got.tags[0].redacted);
        assert_ne!(
            got.tags[0].tag, before[0],
            "the credential must not reach the ledger"
        );
        assert_eq!(got.tags[0].source, TagSource::Caller);
        assert!(!got.tags[1].redacted);
        assert_eq!(got.redacted.key, 1);
    }

    #[test]
    fn first_occurrence_wins_so_authored_beats_derived() {
        let sources = [TagSource::Caller, TagSource::Ner];
        let before = s(&["veld", "veld"]);
        let got = TagSources::label(
            &sources,
            &before,
            &before,
            &s(&["veld"]),
            &RedactionReport::default(),
        );
        assert_eq!(got.tags[0].source, TagSource::Caller);
    }

    #[test]
    fn reconcile_unions_a_merge_and_refuses_an_unknown_tag() {
        let r = RedactionReport::default();
        let incoming = TagSources::all_caller(&s(&["new"]), &s(&["new"]), &r);
        let existing = TagSources::label(
            &[TagSource::Ner],
            &s(&["old"]),
            &s(&["old"]),
            &s(&["old"]),
            &r,
        );
        let merged = TagSources::reconcile(
            incoming.clone(),
            Some(existing.clone()),
            &s(&["old", "new"]),
        )
        .unwrap();
        assert_eq!(merged.tags[0].source, TagSource::Ner);
        assert_eq!(merged.tags[1].source, TagSource::Caller);

        assert_eq!(
            TagSources::reconcile(incoming.clone(), Some(existing), &s(&["old", "new", "x"])),
            None,
            "a stored tag nobody labelled makes the record undetermined"
        );
        assert_eq!(
            TagSources::reconcile(incoming.clone(), None, &s(&["new"])),
            Some(incoming)
        );
    }

    #[test]
    fn sources_serialise_as_the_agreed_vocabulary() {
        let all = [
            TagSource::Caller,
            TagSource::Intent,
            TagSource::Ingest,
            TagSource::Ner,
            TagSource::Yake,
            TagSource::ColdStart,
        ];
        let json: Vec<String> = all
            .iter()
            .map(|s| serde_json::to_string(s).unwrap())
            .collect();
        assert_eq!(
            json,
            [
                "\"caller\"",
                "\"intent\"",
                "\"ingest\"",
                "\"ner\"",
                "\"yake\"",
                "\"cold-start\""
            ]
        );
        for s in all {
            assert_eq!(
                serde_json::to_string(&s).unwrap(),
                format!("\"{}\"", s.as_str())
            );
        }
    }

    #[test]
    fn store_round_trips_replaces_and_fails_closed_on_both_engines() {
        for (backend, kv, _tmp) in key_value_stores("tag_sources_test") {
            let store = TagSourceStore::with_kv(kv.clone());
            let id = MemoryId(uuid::Uuid::new_v4());
            assert_eq!(store.get(&id).unwrap(), None, "{backend:?}: absent is None");

            let first = TagSources::all_caller(&s(&["a"]), &s(&["a"]), &RedactionReport::default());
            store.put(&id, &first).unwrap();
            let second =
                TagSources::all_caller(&s(&["b"]), &s(&["b"]), &RedactionReport::default());
            store.put(&id, &second).unwrap();
            assert_eq!(
                store.get(&id).unwrap(),
                Some(second.clone()),
                "{backend:?}: replaced"
            );
            assert_eq!(store.all().unwrap().len(), 1, "{backend:?}");

            kv.put_bytes(
                format!("{TAG_SOURCE_PREFIX}{}", uuid::Uuid::new_v4()).as_bytes(),
                b"not json",
            )
            .unwrap();
            assert!(
                store.all().is_err(),
                "{backend:?}: a corrupt row is an error, not a gap"
            );

            store.delete(&id).unwrap();
            assert_eq!(store.get(&id).unwrap(), None, "{backend:?}: deleted");
        }
    }
}
