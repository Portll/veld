//! Entity mentions per time bucket, computed from episodes.
//!
//! `mention_count` on an entity is all-time. This module answers "how often was X mentioned
//! this week vs last" by walking episodes and their `entity_refs`, bucketed on the episode's
//! event time. Everything is pure and deterministic: BTreeMaps throughout, names sorted, and
//! the window is a parameter rather than the clock.

use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Datelike, Duration, NaiveTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entity_hygiene::{classify, Shelf};
use crate::graph_memory::EpisodicNode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Bucket {
    #[default]
    Day,
    Week,
}

impl Bucket {
    pub fn start_of(self, at: DateTime<Utc>) -> DateTime<Utc> {
        let date = match self {
            Bucket::Day => at.date_naive(),
            Bucket::Week => {
                at.date_naive() - Duration::days(at.weekday().num_days_from_monday() as i64)
            }
        };
        Utc.from_utc_datetime(&date.and_time(NaiveTime::MIN))
    }
}

#[derive(Debug, Clone)]
pub struct Window {
    pub since: DateTime<Utc>,
    pub until: DateTime<Utc>,
    pub bucket: Bucket,
    pub limit: usize,
    pub shelf: Option<Shelf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BucketCounts {
    pub start: DateTime<Utc>,
    pub counts: BTreeMap<String, usize>,
}

/// Why the result cannot be read as a measurement of the window. Absent when it can.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Undetermined {
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Frequency {
    pub since: DateTime<Utc>,
    pub until: DateTime<Utc>,
    pub bucket: Bucket,
    pub witness: &'static str,
    pub episodes_total: usize,
    pub episodes_in_window: usize,
    pub episodes_undecodable: usize,
    pub refs_unresolved: usize,
    pub entities_in_window: usize,
    pub buckets: Vec<BucketCounts>,
    pub totals: BTreeMap<String, usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub undetermined: Option<Undetermined>,
}

/// `names` resolves an entity uuid to its name and proper-noun flag; a ref it cannot resolve is
/// counted in `refs_unresolved` rather than dropped silently.
pub fn frequency(
    episodes: &[EpisodicNode],
    undecodable: usize,
    names: &HashMap<Uuid, (String, bool)>,
    window: &Window,
) -> Frequency {
    let mut per_bucket: BTreeMap<DateTime<Utc>, BTreeMap<String, usize>> = BTreeMap::new();
    let mut totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut refs_unresolved = 0usize;
    let mut in_window = 0usize;

    for episode in episodes {
        if episode.valid_at < window.since || episode.valid_at >= window.until {
            continue;
        }
        in_window += 1;
        let start = window.bucket.start_of(episode.valid_at);
        for entity in &episode.entity_refs {
            let Some((name, proper)) = names.get(entity) else {
                refs_unresolved += 1;
                continue;
            };
            if let Some(shelf) = window.shelf {
                if classify(name, *proper).shelf != shelf {
                    continue;
                }
            }
            *per_bucket.entry(start).or_default().entry(name.clone()).or_default() += 1;
            *totals.entry(name.clone()).or_default() += 1;
        }
    }

    let mut ranked: Vec<(&String, &usize)> = totals.iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let kept: Vec<String> = ranked.iter().take(window.limit).map(|(n, _)| (*n).clone()).collect();
    let entities_in_window = totals.len();
    let totals: BTreeMap<String, usize> =
        kept.iter().map(|n| (n.clone(), totals[n])).collect();
    let buckets = per_bucket
        .into_iter()
        .map(|(start, counts)| BucketCounts {
            start,
            counts: counts.into_iter().filter(|(n, _)| totals.contains_key(n)).collect(),
        })
        .collect();

    let undetermined = if undecodable > 0 {
        Some(Undetermined {
            reason: format!("{undecodable} episode rows could not be decoded; counts are a lower bound"),
        })
    } else if in_window == 0 {
        Some(Undetermined {
            reason: "no episodes carry an event time inside the window".to_string(),
        })
    } else {
        None
    };

    Frequency {
        since: window.since,
        until: window.until,
        bucket: window.bucket,
        witness: "episodes.valid_at × entity_refs",
        episodes_total: episodes.len(),
        episodes_in_window: in_window,
        episodes_undecodable: undecodable,
        refs_unresolved,
        entities_in_window,
        buckets,
        totals,
        undetermined,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_memory::EpisodeSource;

    fn at(s: &str) -> DateTime<Utc> {
        s.parse().unwrap()
    }

    fn episode(valid_at: &str, refs: &[Uuid]) -> EpisodicNode {
        EpisodicNode {
            uuid: Uuid::new_v4(),
            name: String::new(),
            content: String::new(),
            valid_at: at(valid_at),
            created_at: at("2026-12-31T00:00:00Z"),
            entity_refs: refs.to_vec(),
            source: EpisodeSource::Event,
            metadata: HashMap::new(),
        }
    }

    type Fixture = (Vec<EpisodicNode>, HashMap<Uuid, (String, bool)>, Uuid, Uuid, Uuid);

    fn fixture() -> Fixture {
        let alpha = Uuid::from_u128(1);
        let stamp = Uuid::from_u128(2);
        let ghost = Uuid::from_u128(3);
        let names = HashMap::from([
            (alpha, ("alpha".to_string(), false)),
            (stamp, ("2026-01-01T00:00:00".to_string(), false)),
        ]);
        let episodes = vec![
            episode("2026-03-02T10:00:00Z", &[alpha, stamp]),
            episode("2026-03-03T10:00:00Z", &[alpha]),
            episode("2026-03-09T10:00:00Z", &[alpha, ghost]),
            episode("2026-02-01T10:00:00Z", &[alpha]),
        ];
        (episodes, names, alpha, stamp, ghost)
    }

    fn window(bucket: Bucket, shelf: Option<Shelf>) -> Window {
        Window {
            since: at("2026-03-01T00:00:00Z"),
            until: at("2026-03-15T00:00:00Z"),
            bucket,
            limit: 10,
            shelf,
        }
    }

    #[test]
    fn day_buckets_count_mentions_inside_the_window_only() {
        let (episodes, names, ..) = fixture();
        let f = frequency(&episodes, 0, &names, &window(Bucket::Day, None));
        assert_eq!(f.episodes_total, 4);
        assert_eq!(f.episodes_in_window, 3);
        assert_eq!(f.refs_unresolved, 1);
        assert_eq!(f.totals["alpha"], 3);
        assert_eq!(f.totals["2026-01-01T00:00:00"], 1);
        assert_eq!(f.buckets.len(), 3);
        assert_eq!(f.buckets[0].start, at("2026-03-02T00:00:00Z"));
        assert!(f.undetermined.is_none());
    }

    #[test]
    fn week_buckets_start_on_monday() {
        let (episodes, names, ..) = fixture();
        let f = frequency(&episodes, 0, &names, &window(Bucket::Week, None));
        let starts: Vec<_> = f.buckets.iter().map(|b| b.start).collect();
        assert_eq!(starts, vec![at("2026-03-02T00:00:00Z"), at("2026-03-09T00:00:00Z")]);
        assert_eq!(f.buckets[0].counts["alpha"], 2);
        assert_eq!(f.buckets[1].counts["alpha"], 1);
    }

    #[test]
    fn shelf_filter_drops_noise_entities() {
        let (episodes, names, ..) = fixture();
        let f = frequency(&episodes, 0, &names, &window(Bucket::Day, Some(Shelf::Vocabulary)));
        assert!(f.totals.contains_key("alpha"));
        assert!(!f.totals.contains_key("2026-01-01T00:00:00"));
        assert_eq!(f.entities_in_window, 1);
    }

    #[test]
    fn limit_keeps_the_top_names_and_reports_the_full_count() {
        let (episodes, names, ..) = fixture();
        let mut w = window(Bucket::Day, None);
        w.limit = 1;
        let f = frequency(&episodes, 0, &names, &w);
        assert_eq!(f.totals.len(), 1);
        assert!(f.totals.contains_key("alpha"));
        assert_eq!(f.entities_in_window, 2);
        assert!(f.buckets.iter().all(|b| !b.counts.contains_key("2026-01-01T00:00:00")));
    }

    #[test]
    fn empty_window_is_undetermined_not_zero() {
        let (episodes, names, ..) = fixture();
        let mut w = window(Bucket::Day, None);
        w.since = at("2027-01-01T00:00:00Z");
        w.until = at("2027-02-01T00:00:00Z");
        let f = frequency(&episodes, 0, &names, &w);
        assert_eq!(f.episodes_in_window, 0);
        assert!(f.totals.is_empty());
        assert!(f.undetermined.is_some());
    }

    #[test]
    fn undecodable_rows_make_the_count_a_lower_bound() {
        let (episodes, names, ..) = fixture();
        let f = frequency(&episodes, 2, &names, &window(Bucket::Day, None));
        assert_eq!(f.episodes_undecodable, 2);
        assert!(f.undetermined.unwrap().reason.contains("lower bound"));
    }

    #[test]
    fn same_inputs_same_bytes() {
        let (episodes, names, ..) = fixture();
        let a = serde_json::to_string(&frequency(&episodes, 0, &names, &window(Bucket::Day, None))).unwrap();
        let b = serde_json::to_string(&frequency(&episodes, 0, &names, &window(Bucket::Day, None))).unwrap();
        assert_eq!(a, b);
    }
}
