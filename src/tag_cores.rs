//! Stamped tag cores: tags that sit on exactly the same records.
//!
//! A producer that emits the same boilerplate into every record it writes (a session-end hook,
//! a rollup) leaves a signature no lexical rule can see: a dozen ordinary words that always
//! appear together on the same memories. The census groups tags by the exact set of memories
//! carrying them, read from the tag-source store, and reports every group large enough in both
//! directions. Detection is by identical record membership, not identical tag sets: two tags
//! can share every record and still have different co-tags elsewhere.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::Serialize;

use crate::memory::tag_sources::TagSources;
use crate::memory::MemoryId;

#[derive(Debug, Clone, Copy)]
pub struct CoreThresholds {
    pub min_records: usize,
    pub min_tags: usize,
    pub limit: usize,
}

impl Default for CoreThresholds {
    fn default() -> Self {
        Self {
            min_records: 10,
            min_tags: 3,
            limit: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct StampedCore {
    pub records: usize,
    pub tags: Vec<String>,
    /// How many of the tag placements in this core came from each source.
    pub sources: BTreeMap<&'static str, usize>,
    pub redacted_placements: usize,
    pub sample_memory_ids: Vec<MemoryId>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Undetermined {
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct StampedCores {
    pub witness: &'static str,
    pub rows_seen: usize,
    pub distinct_tags: usize,
    pub min_records: usize,
    pub min_tags: usize,
    pub cores_found: usize,
    pub cores: Vec<StampedCore>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub undetermined: Option<Undetermined>,
}

pub fn stamped_cores(rows: &HashMap<MemoryId, TagSources>, t: CoreThresholds) -> StampedCores {
    let mut membership: BTreeMap<String, BTreeSet<MemoryId>> = BTreeMap::new();
    for (id, sources) in rows {
        for tag in &sources.tags {
            membership.entry(tag.tag.clone()).or_default().insert(id.clone());
        }
    }
    let distinct_tags = membership.len();

    let mut by_set: BTreeMap<Vec<MemoryId>, Vec<String>> = BTreeMap::new();
    for (tag, set) in membership {
        if set.len() < t.min_records {
            continue;
        }
        by_set.entry(set.into_iter().collect()).or_default().push(tag);
    }

    let mut cores: Vec<StampedCore> = by_set
        .into_iter()
        .filter(|(_, tags)| tags.len() >= t.min_tags)
        .map(|(members, tags)| {
            let mut sources: BTreeMap<&'static str, usize> = BTreeMap::new();
            let mut redacted = 0usize;
            for id in &members {
                for st in &rows[id].tags {
                    if tags.binary_search(&st.tag).is_ok() {
                        *sources.entry(st.source.as_str()).or_default() += 1;
                        if st.redacted {
                            redacted += 1;
                        }
                    }
                }
            }
            StampedCore {
                records: members.len(),
                tags,
                sources,
                redacted_placements: redacted,
                sample_memory_ids: members.iter().take(3).cloned().collect(),
            }
        })
        .collect();
    cores.sort_by(|a, b| {
        b.records
            .cmp(&a.records)
            .then_with(|| b.tags.len().cmp(&a.tags.len()))
            .then_with(|| a.tags.cmp(&b.tags))
    });
    let cores_found = cores.len();
    cores.truncate(t.limit);

    let undetermined = if rows.is_empty() {
        Some(Undetermined {
            reason: "the tag-source store has no rows; records written before it existed carry no provenance and cannot be censused".to_string(),
        })
    } else {
        None
    };

    StampedCores {
        witness: "tag_source_store.all() grouped by exact memory membership",
        rows_seen: rows.len(),
        distinct_tags,
        min_records: t.min_records,
        min_tags: t.min_tags,
        cores_found,
        cores,
        undetermined,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::tag_sources::{RedactedCounts, SourcedTag, TagSource, TAG_SOURCES_SCHEMA};
    use uuid::Uuid;

    fn id(n: u128) -> MemoryId {
        MemoryId(Uuid::from_u128(n))
    }

    fn row(tags: &[(&str, TagSource)]) -> TagSources {
        TagSources {
            v: TAG_SOURCES_SCHEMA,
            tags: tags
                .iter()
                .map(|(t, s)| SourcedTag {
                    tag: t.to_string(),
                    source: *s,
                    redacted: false,
                })
                .collect(),
            redacted: RedactedCounts::default(),
        }
    }

    fn thresholds(min_records: usize, min_tags: usize) -> CoreThresholds {
        CoreThresholds {
            min_records,
            min_tags,
            limit: 10,
        }
    }

    /// Twelve records stamped with the same three boilerplate tags by a hook, each also carrying
    /// one tag of its own; a thirteenth record shares two of the three but not the third.
    fn fixture() -> HashMap<MemoryId, TagSources> {
        let mut rows = HashMap::new();
        for n in 1..=12u128 {
            let own = format!("topic-{n}");
            rows.insert(
                id(n),
                row(&[
                    ("Session", TagSource::Ingest),
                    ("Persisted", TagSource::Ingest),
                    ("NOT", TagSource::Yake),
                    (own.as_str(), TagSource::Caller),
                ]),
            );
        }
        rows.insert(
            id(13),
            row(&[("Session", TagSource::Ingest), ("Persisted", TagSource::Ingest)]),
        );
        rows
    }

    #[test]
    fn a_core_is_tags_with_identical_record_membership() {
        let out = stamped_cores(&fixture(), thresholds(10, 1));
        assert_eq!(out.rows_seen, 13);
        assert_eq!(out.cores_found, 2);
        let big = &out.cores[0];
        assert_eq!(big.records, 13);
        assert_eq!(big.tags, vec!["Persisted".to_string(), "Session".to_string()]);
        let three = &out.cores[1];
        assert_eq!(three.records, 12);
        assert_eq!(three.tags, vec!["NOT".to_string()]);
        assert!(out.undetermined.is_none());
    }

    #[test]
    fn min_tags_drops_lone_tags_even_when_widely_shared() {
        let two = stamped_cores(&fixture(), thresholds(10, 2));
        assert_eq!(two.cores_found, 1, "NOT sits alone on its membership and is dropped");
        assert_eq!(two.cores[0].tags.len(), 2);
        let three = stamped_cores(&fixture(), thresholds(10, 3));
        assert_eq!(three.cores_found, 0, "no three tags share one exact membership here");
    }

    #[test]
    fn min_records_drops_small_groups() {
        let out = stamped_cores(&fixture(), thresholds(14, 1));
        assert_eq!(out.cores_found, 0);
        assert_eq!(out.distinct_tags, 15);
    }

    #[test]
    fn sources_are_tallied_per_placement() {
        let out = stamped_cores(&fixture(), thresholds(10, 2));
        let big = &out.cores[0];
        assert_eq!(big.sources["ingest"], 26);
        assert_eq!(big.redacted_placements, 0);
        assert_eq!(big.sample_memory_ids.len(), 3);
    }

    #[test]
    fn an_empty_store_is_undetermined_not_clean() {
        let out = stamped_cores(&HashMap::new(), CoreThresholds::default());
        assert_eq!(out.cores_found, 0);
        assert!(out.undetermined.unwrap().reason.contains("no rows"));
    }

    #[test]
    fn limit_truncates_but_reports_the_full_count() {
        let mut t = thresholds(10, 1);
        t.limit = 1;
        let out = stamped_cores(&fixture(), t);
        assert_eq!(out.cores_found, 2);
        assert_eq!(out.cores.len(), 1);
    }

    #[test]
    fn same_inputs_same_bytes() {
        let f = fixture();
        let a = serde_json::to_string(&stamped_cores(&f, thresholds(10, 2))).unwrap();
        let b = serde_json::to_string(&stamped_cores(&f, thresholds(10, 2))).unwrap();
        assert_eq!(a, b);
    }
}
