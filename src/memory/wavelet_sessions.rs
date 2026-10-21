//! Gap-Based Session Detection
//!
//! Detects natural session boundaries from memory creation timestamps using
//! inter-memory gap thresholding. The fixed-threshold variant splits at gaps over
//! `SESSION_GAP_THRESHOLD_SECS`; the adaptive variant splits at the largest
//! relative jump in the sorted gap distribution, searched inside a plausible
//! session-break window.
//!
//! This enables ordinal resolution: "second meeting" -> session[1]'s memories.

use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::constants::SESSION_GAP_THRESHOLD_SECS;
use crate::memory::MemoryId;

/// A single detected session — a contiguous group of memories separated by
/// large temporal gaps from neighboring sessions.
#[derive(Debug, Clone)]
pub struct DetectedSession {
    /// 1-based ordinal of this session in chronological order.
    pub ordinal: usize,
    /// Memory IDs belonging to this session, sorted by creation time.
    pub memory_ids: Vec<MemoryId>,
    /// Earliest memory timestamp in this session.
    pub start: DateTime<Utc>,
    /// Latest memory timestamp in this session.
    pub end: DateTime<Utc>,
    /// Number of memories in this session.
    pub count: usize,
}

/// Statistics from the adaptive session detection algorithm.
///
/// Reports what the gap analysis computed, including whether the adaptive
/// threshold was used or the algorithm fell back to the fixed
/// `SESSION_GAP_THRESHOLD_SECS` constant.
#[derive(Debug, Clone)]
pub struct SessionDetectionStats {
    /// The threshold (in seconds) used to split sessions.
    pub threshold_secs: i64,
    /// `true` if the adaptive partition-gap threshold was used; `false` if
    /// the algorithm fell back to the fixed constant.
    pub adaptive: bool,
    /// Number of inter-memory gaps in the input.
    pub gap_count: usize,
    /// Median inter-memory gap in seconds.
    pub median_gap_secs: f64,
    /// Index in the sorted gap array of the last within-session gap: the split
    /// falls between it and the next gap. Only meaningful when `adaptive` is `true`.
    pub partition_index: usize,
    /// Relative jump `(hi - lo) / lo` across the split, both gaps clamped to
    /// [300 s, 28 800 s]. Only meaningful when `adaptive` is `true`.
    pub relative_jump: f64,
}

/// Computed session map over the entire memory store.
#[derive(Debug, Clone)]
pub struct SessionMap {
    /// Sessions in chronological order (index 0 = earliest session).
    pub sessions: Vec<DetectedSession>,
    /// Reverse lookup: memory ID -> session ordinal (1-based).
    pub memory_to_session: HashMap<MemoryId, usize>,
    /// When this map was computed.
    pub computed_at: DateTime<Utc>,
}

/// Detect session boundaries from memory timestamps.
///
/// Sorts memories by creation time, then walks the sorted list looking for
/// inter-memory gaps exceeding `SESSION_GAP_THRESHOLD_SECS` (7200s = 2h).
/// Each gap marks a session boundary. O(n log n) due to the initial sort.
pub fn detect_sessions(memories: &[(MemoryId, DateTime<Utc>)]) -> SessionMap {
    if memories.is_empty() {
        return SessionMap {
            sessions: Vec::new(),
            memory_to_session: HashMap::new(),
            computed_at: Utc::now(),
        };
    }

    // Sort by timestamp ascending
    let mut sorted: Vec<(MemoryId, DateTime<Utc>)> = memories.to_vec();
    sorted.sort_by_key(|(_, ts)| *ts);

    let mut sessions: Vec<DetectedSession> = Vec::new();
    let mut memory_to_session: HashMap<MemoryId, usize> = HashMap::new();

    // Start the first session
    let mut current_ids: Vec<MemoryId> = vec![sorted[0].0.clone()];
    let mut current_start = sorted[0].1;
    let mut current_end = sorted[0].1;

    for window in sorted.windows(2) {
        let (_prev_id, prev_ts) = &window[0];
        let (cur_id, cur_ts) = &window[1];

        let gap_secs = (*cur_ts - *prev_ts).num_seconds();

        if gap_secs > SESSION_GAP_THRESHOLD_SECS {
            // Close current session
            let ordinal = sessions.len() + 1;
            let count = current_ids.len();
            for id in &current_ids {
                memory_to_session.insert(id.clone(), ordinal);
            }
            sessions.push(DetectedSession {
                ordinal,
                memory_ids: current_ids,
                start: current_start,
                end: current_end,
                count,
            });

            // Start new session
            current_ids = vec![cur_id.clone()];
            current_start = *cur_ts;
            current_end = *cur_ts;
        } else {
            current_ids.push(cur_id.clone());
            current_end = *cur_ts;
        }
    }

    // Close final session
    let ordinal = sessions.len() + 1;
    let count = current_ids.len();
    for id in &current_ids {
        memory_to_session.insert(id.clone(), ordinal);
    }
    sessions.push(DetectedSession {
        ordinal,
        memory_ids: current_ids,
        start: current_start,
        end: current_end,
        count,
    });

    SessionMap {
        sessions,
        memory_to_session,
        computed_at: Utc::now(),
    }
}

/// Minimum adaptive threshold (seconds). A pause shorter than 5 minutes is not
/// a meaningful interaction boundary, so gaps below it are searched as if they
/// were exactly 5 minutes long.
const ADAPTIVE_THRESHOLD_MIN_SECS: f64 = 300.0;

/// Maximum adaptive threshold (seconds). Gaps beyond 8 hours almost certainly
/// separate sessions whatever the distribution, so gaps above it are searched
/// as if they were exactly 8 hours long.
const ADAPTIVE_THRESHOLD_MAX_SECS: f64 = 28800.0;

/// Minimum number of inter-memory gaps required to run the adaptive
/// algorithm. Below this count, the gap distribution is too sparse for
/// reliable partition detection, so we fall back to the fixed threshold.
const ADAPTIVE_MIN_GAPS: usize = 10;

/// Detect session boundaries with a threshold adapted to the gap distribution.
///
/// Instead of the fixed 7200s gap used by [`detect_sessions`], this function
/// sorts the inter-memory gaps and splits them where consecutive sorted gaps
/// jump the most, relative to the smaller: everything at or below the jump is
/// within a session, everything above it separates two. The search clamps each
/// gap to [300 s, 28 800 s] first (see [`partition_split`]), and the threshold is
/// the geometric mean of the clamped pair.
///
/// Falls back to `SESSION_GAP_THRESHOLD_SECS` when:
/// - Fewer than 10 inter-memory gaps exist (distribution too sparse)
/// - No two gaps differ once clamped to the window (nothing to split on)
///
/// Returns both the `SessionMap` and a `SessionDetectionStats` struct
/// documenting the algorithm's decisions.
pub fn detect_sessions_adaptive(
    memories: &[(MemoryId, DateTime<Utc>)],
) -> (SessionMap, SessionDetectionStats) {
    if memories.is_empty() {
        let map = SessionMap {
            sessions: Vec::new(),
            memory_to_session: HashMap::new(),
            computed_at: Utc::now(),
        };
        let stats = SessionDetectionStats {
            threshold_secs: SESSION_GAP_THRESHOLD_SECS,
            adaptive: false,
            gap_count: 0,
            median_gap_secs: 0.0,
            partition_index: 0,
            relative_jump: 0.0,
        };
        return (map, stats);
    }

    // Sort by timestamp ascending
    let mut sorted: Vec<(MemoryId, DateTime<Utc>)> = memories.to_vec();
    sorted.sort_by_key(|(_, ts)| *ts);

    // Compute inter-memory gaps
    let gaps: Vec<i64> = sorted
        .windows(2)
        .map(|w| (w[1].1 - w[0].1).num_seconds())
        .collect();

    let gap_count = gaps.len();

    let mut sorted_gaps = gaps.clone();
    sorted_gaps.sort_unstable();

    let median_gap_secs = if sorted_gaps.is_empty() {
        0.0
    } else if sorted_gaps.len().is_multiple_of(2) {
        let mid = sorted_gaps.len() / 2;
        (sorted_gaps[mid - 1] + sorted_gaps[mid]) as f64 / 2.0
    } else {
        sorted_gaps[sorted_gaps.len() / 2] as f64
    };

    let split = if gap_count < ADAPTIVE_MIN_GAPS {
        None
    } else {
        partition_split(&sorted_gaps)
    };
    let (threshold_secs, adaptive, partition_index, relative_jump) = match split {
        Some(split) => (split.threshold_secs, true, split.index, split.relative_jump),
        None => (SESSION_GAP_THRESHOLD_SECS, false, 0, 0.0),
    };

    // Build session map using the computed threshold
    let mut sessions: Vec<DetectedSession> = Vec::new();
    let mut memory_to_session: HashMap<MemoryId, usize> = HashMap::new();

    let mut current_ids: Vec<MemoryId> = vec![sorted[0].0.clone()];
    let mut current_start = sorted[0].1;
    let mut current_end = sorted[0].1;

    for (i, window) in sorted.windows(2).enumerate() {
        let (cur_id, cur_ts) = &window[1];
        let gap = gaps[i];

        if gap > threshold_secs {
            let ordinal = sessions.len() + 1;
            let count = current_ids.len();
            for id in &current_ids {
                memory_to_session.insert(id.clone(), ordinal);
            }
            sessions.push(DetectedSession {
                ordinal,
                memory_ids: current_ids,
                start: current_start,
                end: current_end,
                count,
            });

            current_ids = vec![cur_id.clone()];
            current_start = *cur_ts;
            current_end = *cur_ts;
        } else {
            current_ids.push(cur_id.clone());
            current_end = *cur_ts;
        }
    }

    // Close final session
    let ordinal = sessions.len() + 1;
    let count = current_ids.len();
    for id in &current_ids {
        memory_to_session.insert(id.clone(), ordinal);
    }
    sessions.push(DetectedSession {
        ordinal,
        memory_ids: current_ids,
        start: current_start,
        end: current_end,
        count,
    });

    let map = SessionMap {
        sessions,
        memory_to_session,
        computed_at: Utc::now(),
    };

    let stats = SessionDetectionStats {
        threshold_secs,
        adaptive,
        gap_count,
        median_gap_secs,
        partition_index,
        relative_jump,
    };

    (map, stats)
}

/// Where [`detect_sessions_adaptive`] splits a sorted gap sequence.
struct PartitionSplit {
    /// Index of the last gap below the split.
    index: usize,
    /// `(hi - lo) / lo` across the split, on the clamped gaps.
    relative_jump: f64,
    /// Gaps strictly above this separate sessions.
    threshold_secs: i64,
}

/// The largest relative jump between consecutive gaps of `sorted_gaps`, each
/// clamped to [`ADAPTIVE_THRESHOLD_MIN_SECS`, `ADAPTIVE_THRESHOLD_MAX_SECS`], and
/// the threshold for it: the geometric mean of the clamped pair, floored.
/// `None` when no pair differs once clamped.
///
/// Clamping before comparing is what keeps out-of-window noise from winning. On
/// raw gaps a burst of 1 s writes beside 30 s ones is a relative jump of 29, which
/// outranks the 11 from 5 minutes to an hour and puts the split far below any
/// session break; a week-long absence beside a 2 h break does the same at the top.
/// Clamped, every gap under 5 minutes reads as 5 minutes and every gap over 8
/// hours as 8 hours, so those pairs jump by nothing.
///
/// The floored geometric mean of a clamped pair `lo < hi` lies in `[lo, hi)`, and
/// a positive jump means the raw pair is at least that far apart (raw `lo` no
/// larger, raw `hi` no smaller), so the threshold keeps exactly the gaps up to
/// the chosen index inside sessions. Any value in that interval gives the same
/// partition; the geometric mean is reported because the jump is measured as a
/// ratio, and on that scale it sits midway between the pair.
fn partition_split(sorted_gaps: &[i64]) -> Option<PartitionSplit> {
    let clamp =
        |gap: i64| (gap as f64).clamp(ADAPTIVE_THRESHOLD_MIN_SECS, ADAPTIVE_THRESHOLD_MAX_SECS);
    let mut best: Option<(usize, f64)> = None;
    for (index, pair) in sorted_gaps.windows(2).enumerate() {
        let (lo, hi) = (clamp(pair[0]), clamp(pair[1]));
        let jump = (hi - lo) / lo;
        if jump > best.map_or(0.0, |(_, best_jump)| best_jump) {
            best = Some((index, jump));
        }
    }
    let (index, relative_jump) = best?;
    let (lo, hi) = (clamp(sorted_gaps[index]), clamp(sorted_gaps[index + 1]));
    Some(PartitionSplit {
        index,
        relative_jump,
        threshold_secs: (lo * hi).sqrt().floor() as i64,
    })
}

/// Parse an English ordinal word or numeric ordinal (e.g., "1st") to a 1-based index.
///
/// Supports "first" through "tenth" and numeric patterns like "1st", "2nd", "3rd", "4th"..."10th".
pub fn parse_ordinal(word: &str) -> Option<usize> {
    let lower = word.to_lowercase();
    match lower.as_str() {
        "first" => Some(1),
        "second" => Some(2),
        "third" => Some(3),
        "fourth" => Some(4),
        "fifth" => Some(5),
        "sixth" => Some(6),
        "seventh" => Some(7),
        "eighth" => Some(8),
        "ninth" => Some(9),
        "tenth" => Some(10),
        _ => {
            // Try numeric ordinals: "1st", "2nd", "3rd", "4th", etc.
            let stripped = lower
                .strip_suffix("st")
                .or_else(|| lower.strip_suffix("nd"))
                .or_else(|| lower.strip_suffix("rd"))
                .or_else(|| lower.strip_suffix("th"))?;
            stripped.parse::<usize>().ok().filter(|&n| n >= 1)
        }
    }
}

/// Session-type nouns that can follow an ordinal to form a session reference.
const SESSION_NOUNS: &[&str] = &[
    "meeting",
    "session",
    "sprint",
    "conversation",
    "discussion",
    "phase",
    "call",
    "standup",
    "review",
    "retro",
];

/// Extract an ordinal session reference from a query string.
///
/// Scans for patterns like `{ordinal} {session_noun}` where the ordinal is a
/// word ("second") or numeric ("2nd") and the noun is one of the recognized
/// session-type nouns.
///
/// Returns `(ordinal_1based, matched_noun)` if found.
pub fn extract_ordinal_session_ref(query: &str) -> Option<(usize, String)> {
    let lower = query.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    for pair in words.windows(2) {
        if let Some(ordinal) = parse_ordinal(pair[0]) {
            // Strip trailing punctuation from the noun candidate
            let noun_candidate = pair[1].trim_end_matches(|c: char| !c.is_alphanumeric());
            for noun in SESSION_NOUNS {
                if noun_candidate == *noun {
                    return Some((ordinal, noun.to_string()));
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use uuid::Uuid;

    /// Helper: create a (MemoryId, DateTime) pair at a given offset from a base time.
    fn mem_at(base: DateTime<Utc>, offset_secs: i64) -> (MemoryId, DateTime<Utc>) {
        let ts = base + chrono::Duration::seconds(offset_secs);
        (MemoryId(Uuid::new_v4()), ts)
    }

    #[test]
    fn test_detect_four_sessions() {
        // 4 groups of 15 memories each, separated by 1-week gaps.
        let base = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        let week_secs: i64 = 7 * 24 * 3600;
        let mut memories = Vec::new();

        for session_idx in 0..4u32 {
            let session_base = session_idx as i64 * week_secs;
            for mem_idx in 0..15u32 {
                // Spread within each session: 5 minutes apart
                memories.push(mem_at(base, session_base + mem_idx as i64 * 300));
            }
        }

        let map = detect_sessions(&memories);

        assert_eq!(map.sessions.len(), 4, "Should detect exactly 4 sessions");
        for (i, session) in map.sessions.iter().enumerate() {
            assert_eq!(session.ordinal, i + 1);
            assert_eq!(
                session.count,
                15,
                "Session {} should have 15 memories",
                i + 1
            );
            assert_eq!(session.memory_ids.len(), 15);
        }

        // Every memory should be mapped
        assert_eq!(map.memory_to_session.len(), 60);
    }

    #[test]
    fn test_parse_ordinal() {
        // Word ordinals
        assert_eq!(parse_ordinal("first"), Some(1));
        assert_eq!(parse_ordinal("second"), Some(2));
        assert_eq!(parse_ordinal("third"), Some(3));
        assert_eq!(parse_ordinal("fourth"), Some(4));
        assert_eq!(parse_ordinal("fifth"), Some(5));
        assert_eq!(parse_ordinal("sixth"), Some(6));
        assert_eq!(parse_ordinal("seventh"), Some(7));
        assert_eq!(parse_ordinal("eighth"), Some(8));
        assert_eq!(parse_ordinal("ninth"), Some(9));
        assert_eq!(parse_ordinal("tenth"), Some(10));

        // Numeric ordinals
        assert_eq!(parse_ordinal("1st"), Some(1));
        assert_eq!(parse_ordinal("2nd"), Some(2));
        assert_eq!(parse_ordinal("3rd"), Some(3));
        assert_eq!(parse_ordinal("4th"), Some(4));
        assert_eq!(parse_ordinal("10th"), Some(10));

        // Case insensitivity
        assert_eq!(parse_ordinal("First"), Some(1));
        assert_eq!(parse_ordinal("THIRD"), Some(3));

        // Non-ordinals
        assert_eq!(parse_ordinal("hello"), None);
        assert_eq!(parse_ordinal("0th"), None);
    }

    #[test]
    fn test_extract_ordinal_session_ref() {
        // Positive matches
        assert_eq!(
            extract_ordinal_session_ref("What happened in the second meeting?"),
            Some((2, "meeting".to_string()))
        );
        assert_eq!(
            extract_ordinal_session_ref("third sprint goals"),
            Some((3, "sprint".to_string()))
        );
        assert_eq!(
            extract_ordinal_session_ref("Notes from the 1st standup"),
            Some((1, "standup".to_string()))
        );
        assert_eq!(
            extract_ordinal_session_ref("the fifth discussion"),
            Some((5, "discussion".to_string()))
        );

        // Non-matches
        assert_eq!(
            extract_ordinal_session_ref("What is the weather today?"),
            None
        );
        assert_eq!(
            extract_ordinal_session_ref("second breakfast"), // not a session noun
            None
        );
        assert_eq!(
            extract_ordinal_session_ref("meeting notes"), // no ordinal
            None
        );
    }

    // ---- Adaptive session detection tests ----

    #[test]
    fn test_adaptive_empty_input() {
        let memories: Vec<(MemoryId, DateTime<Utc>)> = Vec::new();
        let (map, stats) = detect_sessions_adaptive(&memories);
        assert!(map.sessions.is_empty());
        assert!(!stats.adaptive);
        assert_eq!(stats.gap_count, 0);
        assert_eq!(stats.threshold_secs, SESSION_GAP_THRESHOLD_SECS);
    }

    #[test]
    fn test_adaptive_fewer_than_10_gaps_falls_back() {
        // 8 memories = 7 gaps, below the ADAPTIVE_MIN_GAPS threshold of 10
        let base = Utc.with_ymd_and_hms(2025, 6, 1, 10, 0, 0).unwrap();
        let mut memories = Vec::new();
        for i in 0..8 {
            memories.push(mem_at(base, i * 600)); // 10 minutes apart
        }

        let (map, stats) = detect_sessions_adaptive(&memories);

        assert!(!stats.adaptive, "Should fall back with fewer than 10 gaps");
        assert_eq!(stats.gap_count, 7);
        assert_eq!(stats.threshold_secs, SESSION_GAP_THRESHOLD_SECS);
        // All memories within 70 minutes, well under the 2h fixed threshold
        assert_eq!(map.sessions.len(), 1, "All memories in one session");
        assert_eq!(map.sessions[0].count, 8);
    }

    #[test]
    fn test_adaptive_bimodal_distribution() {
        // Bimodal: 5 sessions of 5 memories each.
        // Within-session gaps: 300s (5 minutes)
        // Between-session gaps: 10800s (3 hours)
        // The adaptive algorithm should find the split between 300 and 10800.
        let base = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
        let mut memories = Vec::new();
        let mut offset: i64 = 0;

        for session_idx in 0..5 {
            if session_idx > 0 {
                offset += 10800; // 3-hour gap between sessions
            }
            for mem_idx in 0..5 {
                if mem_idx > 0 {
                    offset += 300; // 5-minute gap within session
                }
                memories.push(mem_at(base, offset));
            }
        }

        let (map, stats) = detect_sessions_adaptive(&memories);

        assert_eq!(map.sessions.len(), 5, "Should detect 5 sessions");
        assert!(stats.adaptive, "Should use adaptive threshold");
        for session in &map.sessions {
            assert_eq!(session.count, 5, "Each session should have 5 memories");
        }

        // The threshold should be between 300 and 10800
        assert!(
            stats.threshold_secs > 300 && stats.threshold_secs < 10800,
            "Adaptive threshold {} should be between 300 and 10800",
            stats.threshold_secs
        );

        // Every memory should be mapped
        assert_eq!(map.memory_to_session.len(), 25);
    }

    #[test]
    fn test_adaptive_uniform_gaps_falls_back() {
        // All gaps identical (60s): clamped to the 300s floor they are all equal,
        // so there is no jump to split on and the fixed threshold applies.
        let base = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
        let mut memories = Vec::new();
        for i in 0..20 {
            memories.push(mem_at(base, i * 60));
        }

        let (map, stats) = detect_sessions_adaptive(&memories);

        assert!(
            !stats.adaptive,
            "Uniform gaps have no jump to split on, triggering fallback"
        );
        assert_eq!(stats.gap_count, 19);
        assert_eq!(stats.threshold_secs, SESSION_GAP_THRESHOLD_SECS);
        // All within 19 minutes, under the 2h fixed threshold
        assert_eq!(map.sessions.len(), 1);
    }

    #[test]
    fn test_adaptive_clearly_separated_sessions() {
        // 3 dense clusters separated by 6-hour gaps.
        // Cluster: 20 memories, 120s apart (total span ~38 minutes).
        // Between clusters: 21600s (6 hours).
        let base = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
        let mut memories = Vec::new();
        let mut offset: i64 = 0;

        for cluster_idx in 0..3 {
            if cluster_idx > 0 {
                offset += 21600; // 6-hour gap
            }
            for mem_idx in 0..20 {
                if mem_idx > 0 {
                    offset += 120; // 2-minute gaps within cluster
                }
                memories.push(mem_at(base, offset));
            }
        }

        let (map, stats) = detect_sessions_adaptive(&memories);

        assert_eq!(map.sessions.len(), 3, "Should detect 3 sessions");
        assert!(stats.adaptive, "Should use adaptive threshold");
        assert_eq!(stats.gap_count, 59);

        for session in &map.sessions {
            assert_eq!(session.count, 20);
        }

        // Threshold should be somewhere between 120 and 21600
        assert!(stats.threshold_secs > 120);
        assert!(stats.threshold_secs < 21600);

        // Relative jump should be large (21600/120 = 180x)
        assert!(
            stats.relative_jump > 10.0,
            "Relative jump {} should be very large for bimodal data",
            stats.relative_jump
        );
    }

    #[test]
    fn test_adaptive_single_memory() {
        let base = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
        let memories = vec![mem_at(base, 0)];

        let (map, stats) = detect_sessions_adaptive(&memories);

        assert_eq!(map.sessions.len(), 1);
        assert_eq!(map.sessions[0].count, 1);
        assert!(!stats.adaptive);
        assert_eq!(stats.gap_count, 0);
    }

    /// Gaps for `sessions` sessions with `within` inside each and the
    /// `between[i]` gap before session `i + 1`, as memories from a fixed base.
    fn memories_with_gaps(
        sessions: usize,
        within: &[i64],
        between: &[i64],
    ) -> Vec<(MemoryId, DateTime<Utc>)> {
        let base = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
        let mut offset: i64 = 0;
        let mut memories = vec![mem_at(base, offset)];
        for session_idx in 0..sessions {
            if session_idx > 0 {
                offset += between[session_idx - 1];
                memories.push(mem_at(base, offset));
            }
            for gap in within {
                offset += gap;
                memories.push(mem_at(base, offset));
            }
        }
        memories
    }

    #[test]
    fn test_adaptive_threshold_is_geometric_mean() {
        // 6 sessions of 3 memories, 600s apart inside a session, 7200s between.
        // The split is 600 -> 7200; its geometric mean is sqrt(4_320_000) = 2078.46.
        let memories = memories_with_gaps(6, &[600, 600], &[7200; 5]);

        let (map, stats) = detect_sessions_adaptive(&memories);

        assert!(stats.adaptive);
        assert_eq!(stats.threshold_secs, 2078);
        assert!((stats.relative_jump - 11.0).abs() < 1e-9);
        assert_eq!(map.sessions.len(), 6);
    }

    #[test]
    fn test_adaptive_ignores_sub_window_noise() {
        // 4 sessions an hour apart. Inside each, bursts of 1s writes among
        // 30-240s pauses. Unrestricted, the largest relative jump is 1s -> 30s
        // (29, against 240s -> 3600s at 14); its threshold of ~15s lies below the
        // 300s floor, so the search fell back to the fixed 7200s and kept all
        // four hours in one session.
        let within = [1, 1, 30, 60, 1, 120, 240, 1, 30];
        let memories = memories_with_gaps(4, &within, &[3600; 3]);

        let mut raw: Vec<i64> = memories
            .windows(2)
            .map(|w| (w[1].1 - w[0].1).num_seconds())
            .collect();
        raw.sort_unstable();
        let unrestricted = (0..raw.len() - 1)
            .filter(|&i| raw[i] > 0)
            .max_by(|&a, &b| {
                let jump = |i: usize| (raw[i + 1] - raw[i]) as f64 / raw[i] as f64;
                jump(a).total_cmp(&jump(b))
            })
            .unwrap();
        assert_eq!((raw[unrestricted], raw[unrestricted + 1]), (1, 30));
        assert_eq!(detect_sessions(&memories).sessions.len(), 1);

        let (map, stats) = detect_sessions_adaptive(&memories);

        assert!(
            stats.adaptive,
            "the 240s -> 3600s jump is inside the window"
        );
        assert_eq!(stats.gap_count, 39);
        // Clamped pair 300 -> 3600: jump 11, threshold floor(sqrt(1_080_000)).
        assert_eq!(stats.partition_index, 35);
        assert!((stats.relative_jump - 11.0).abs() < 1e-9);
        assert_eq!(stats.threshold_secs, 1039);
        assert_eq!(map.sessions.len(), 4);
        for session in &map.sessions {
            assert_eq!(session.count, 10);
        }
    }

    #[test]
    fn test_adaptive_ignores_beyond_window_outlier() {
        // 4 sessions of 2-minute gaps, two of them 2h apart and one ten days
        // apart. Unrestricted, 7200s -> 864000s (119) outranks 120s -> 7200s (59)
        // and its threshold lies past 8h, so the search fell back to 7200s, which
        // a 7200s gap does not exceed: only the ten-day gap split.
        let memories = memories_with_gaps(4, &[120; 7], &[7200, 864_000, 7200]);

        let (map, stats) = detect_sessions_adaptive(&memories);

        assert!(stats.adaptive);
        // Clamped pair 300 -> 7200: threshold floor(sqrt(2_160_000)).
        assert_eq!(stats.threshold_secs, 1469);
        assert_eq!(map.sessions.len(), 4);
    }
}
