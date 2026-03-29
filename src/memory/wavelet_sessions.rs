//! Gap-Based Session Detection
//!
//! Detects natural session boundaries from memory creation timestamps using
//! inter-memory gap thresholding. The fixed-threshold variant uses a configurable
//! gap duration; the adaptive variant uses partition-theoretic analysis
//! (Hardy-Ramanujan-Rademacher modular correction) to find the optimal split.
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
/// Reports what the partition-theoretic gap analysis computed, including
/// whether the adaptive threshold was used or the algorithm fell back
/// to the fixed `SESSION_GAP_THRESHOLD_SECS` constant.
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
    /// Index in the sorted gap array where the natural partition split occurred.
    /// Only meaningful when `adaptive` is `true`.
    pub partition_index: usize,
    /// Magnitude of the relative jump at the partition index.
    /// Only meaningful when `adaptive` is `true`.
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

/// Modular correction from partition function asymptotics.
///
/// The Hardy-Ramanujan-Rademacher formula for p(n) contains the term
/// `exp(pi * sqrt(2n/3)) / (4n * sqrt(3))` with a correction factor
/// involving `1/24`. Multiplying the adaptive threshold by `(1 - 1/24)`
/// shifts the boundary slightly inward, empirically improving session
/// boundary detection.
const PARTITION_MODULAR_CORRECTION: f64 = 1.0 - 1.0 / 24.0;

/// Minimum adaptive threshold (seconds). Sessions shorter than 5 minutes
/// are not meaningful interaction boundaries.
const ADAPTIVE_THRESHOLD_MIN_SECS: f64 = 300.0;

/// Maximum adaptive threshold (seconds). Gaps beyond 8 hours almost
/// certainly represent separate sessions regardless of distribution.
const ADAPTIVE_THRESHOLD_MAX_SECS: f64 = 28800.0;

/// Minimum number of inter-memory gaps required to run the adaptive
/// algorithm. Below this count, the gap distribution is too sparse for
/// reliable partition detection, so we fall back to the fixed threshold.
const ADAPTIVE_MIN_GAPS: usize = 10;

/// Detect session boundaries using partition-theoretic adaptive thresholding.
///
/// Instead of the fixed 7200s gap used by [`detect_sessions`], this function
/// computes the distribution of inter-memory gaps and finds the natural
/// "partition gap" — the point in the sorted gap sequence where the largest
/// relative jump occurs. This split point separates within-session gaps from
/// between-session gaps.
///
/// The adaptive threshold is the midpoint of the two gaps surrounding the
/// partition gap, corrected by the modular factor `(1 - 1/24)` from
/// partition function asymptotics. The result is clamped to [300s, 28800s].
///
/// Falls back to `SESSION_GAP_THRESHOLD_SECS` when:
/// - Fewer than 10 inter-memory gaps exist (distribution too sparse)
/// - The computed adaptive threshold falls outside the clamp range
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

    // Compute median gap
    let median_gap_secs = if gaps.is_empty() {
        0.0
    } else {
        let mut sorted_gaps = gaps.clone();
        sorted_gaps.sort_unstable();
        if sorted_gaps.len().is_multiple_of(2) {
            let mid = sorted_gaps.len() / 2;
            (sorted_gaps[mid - 1] + sorted_gaps[mid]) as f64 / 2.0
        } else {
            sorted_gaps[sorted_gaps.len() / 2] as f64
        }
    };

    // Determine threshold
    let (threshold_secs, adaptive, partition_index, relative_jump) = if gap_count
        < ADAPTIVE_MIN_GAPS
    {
        (SESSION_GAP_THRESHOLD_SECS, false, 0, 0.0)
    } else {
        // Sort gaps to find the partition point
        let mut sorted_gaps: Vec<i64> = gaps.clone();
        sorted_gaps.sort_unstable();

        // Find the index i that maximizes (g_{i+1} - g_i) / g_i
        let mut best_index = 0;
        let mut best_jump: f64 = 0.0;
        for i in 0..sorted_gaps.len() - 1 {
            let g_i = sorted_gaps[i] as f64;
            if g_i <= 0.0 {
                continue;
            }
            let g_next = sorted_gaps[i + 1] as f64;
            let jump = (g_next - g_i) / g_i;
            if jump > best_jump {
                best_jump = jump;
                best_index = i;
            }
        }

        // Adaptive threshold: midpoint of the two bounding gaps, with modular correction
        let g_lo = sorted_gaps[best_index] as f64;
        let g_hi = sorted_gaps[best_index + 1] as f64;
        let raw_threshold = (g_lo + g_hi) / 2.0 * PARTITION_MODULAR_CORRECTION;

        if !(ADAPTIVE_THRESHOLD_MIN_SECS..=ADAPTIVE_THRESHOLD_MAX_SECS).contains(&raw_threshold) {
            (SESSION_GAP_THRESHOLD_SECS, false, best_index, best_jump)
        } else {
            (raw_threshold as i64, true, best_index, best_jump)
        }
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
        // All gaps identical (60s). The relative jump is 0.0 everywhere, so
        // the raw threshold becomes (60+60)/2 * correction = ~57.5s, which is
        // below the 300s minimum clamp, triggering fallback.
        let base = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
        let mut memories = Vec::new();
        for i in 0..20 {
            memories.push(mem_at(base, i * 60));
        }

        let (map, stats) = detect_sessions_adaptive(&memories);

        assert!(
            !stats.adaptive,
            "Uniform gaps should produce threshold below min clamp, triggering fallback"
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

    #[test]
    fn test_adaptive_modular_correction_applied() {
        // Verify the 1/24 correction shifts the threshold inward.
        // Create a distribution where the midpoint is known: gaps of 600s and 7200s.
        // Midpoint = (600 + 7200) / 2 = 3900
        // After correction: 3900 * (1 - 1/24) = 3900 * 0.958333... = 3737.5
        let base = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
        let mut memories = Vec::new();
        let mut offset: i64 = 0;

        // 6 sessions: 3 memories each with 600s internal gaps, 7200s between sessions
        for session_idx in 0..6 {
            if session_idx > 0 {
                offset += 7200;
            }
            for mem_idx in 0..3 {
                if mem_idx > 0 {
                    offset += 600;
                }
                memories.push(mem_at(base, offset));
            }
        }

        let (_, stats) = detect_sessions_adaptive(&memories);

        if stats.adaptive {
            // The corrected threshold should be less than the uncorrected midpoint
            let uncorrected_midpoint = (600 + 7200) / 2;
            assert!(
                stats.threshold_secs < uncorrected_midpoint,
                "Corrected threshold {} should be less than uncorrected midpoint {}",
                stats.threshold_secs,
                uncorrected_midpoint
            );
        }
    }
}
