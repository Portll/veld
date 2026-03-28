//! Wavelet-Based Session Detection
//!
//! Detects natural session boundaries from memory creation timestamps using
//! a Haar wavelet-like decomposition. Large inter-memory gaps correspond to
//! high wavelet coefficients at the session-boundary scale, marking boundaries
//! between distinct interaction sessions.
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
            for noun in SESSION_NOUNS {
                if pair[1] == *noun {
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
                session.count, 15,
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
}
