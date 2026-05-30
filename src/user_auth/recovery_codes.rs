//! One-shot recovery codes for password / 2FA reset.
//!
//! Generation: 10 codes per user, 16 base32 characters apiece (80 bits of
//! entropy), formatted in three groups for human transcription
//! (`ABCDE-FGHJK-MNPQRS`). The alphabet is **Crockford base32** as
//! standardised — digits `0`–`9` plus the uppercase letters `A`–`Z`
//! minus the visually-ambiguous `I`, `L`, `O`, `U`. See
//! <https://www.crockford.com/base32.html>.
//!
//! Storage: only Argon2id hashes are ever persisted — never the plaintext —
//! so a database compromise alone can't yield usable recovery codes. The
//! parameters are intentionally lower than the password hasher's: recovery
//! codes carry ~80 bits of entropy on their own, so they need far less work
//! factor to remain unguessable in offline attack scenarios. Lowering t_cost
//! / m_cost keeps the redeem path snappy without weakening security.
//!
//! Redeem semantics: when a code matches an entry, that entry is removed
//! from the user's `recovery_code_hashes` vector. A second redemption with
//! the same code will fail because no matching hash remains.
//!
//! Concurrency: [`generate_batch`] and [`redeem`] are CPU-bound (10 Argon2id
//! hashes per batch, and N verifies per redeem scan). Their async wrappers
//! [`generate_batch_async`] and [`redeem_async`] acquire one slot from the
//! process-global [`crate::user_auth::argon2_pool`] semaphore and run the
//! synchronous work on the blocking-task pool — same back-pressure discipline
//! the password module uses, so a herd of recovery-code redemptions cannot
//! saturate the CPU.

use crate::user_auth::argon2_pool;
use crate::user_auth::AuthError;

use argon2::password_hash::{
    rand_core::OsRng as PwOsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString,
};
use argon2::{Algorithm, Argon2, Params, Version};
use rand::RngCore;

/// Number of codes issued per enrollment / reset cycle.
pub const CODES_PER_BATCH: usize = 10;

/// Number of raw bytes drawn from the RNG per code. 10 bytes → 16 base32
/// chars → 80 bits of effective entropy (`floor(10 * 8 / 5)` chars exactly,
/// no padding required).
const CODE_RAW_BYTES: usize = 10;

/// Crockford base32 alphabet. Digits `0`–`9` followed by the uppercase
/// letters `A`–`Z` with `I`, `L`, `O`, `U` removed for visual hygiene.
/// 32 characters exactly so a 5-bit chunk maps to one symbol.
///
/// Note: this is the canonical Crockford ordering. An earlier draft
/// (pre-W2 hardening) shipped a near-Crockford alphabet that started with
/// the letters and tail-padded with lowercase `ab` to reach 32 entries —
/// that variant accidentally accepted lowercase characters from the OS RNG
/// stream and lost the canonical digit ordering. The B-Audit pass in the
/// W2 breakers backlog replaced it with the standard Crockford table.
const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// Characters explicitly excluded from the alphabet for visual hygiene.
/// Used in tests to guarantee no generated code contains them; the
/// alphabet definition itself is the source-of-truth, this constant just
/// makes the audit easier to read.
#[allow(dead_code)] // referenced from tests; kept as a readable source-of-truth audit list
const AMBIGUOUS_CHARS: &[char] = &['I', 'L', 'O', 'U'];

/// Argon2id parameters used for hashing recovery codes.
///
/// Recovery codes already carry ~80 bits of entropy, so we trade some
/// memory/time cost (vs the primary password hasher) for redeem-path
/// latency. 16 MiB / 2 passes / 1 lane verifies in ~25 ms on a modern
/// laptop core.
fn code_params() -> Params {
    Params::new(16 * 1024, 2, 1, None).expect("recovery code params are statically valid")
}

fn code_hasher() -> Argon2<'static> {
    Argon2::new(Algorithm::Argon2id, Version::V0x13, code_params())
}

/// Encode raw bytes into the Crockford-ish base32 alphabet (no padding).
///
/// Standard 5-bits-per-symbol packing: emit one symbol per 5 input bits,
/// MSB first. For 10 input bytes (80 bits) this yields exactly 16 symbols
/// with no leftover bits — no padding required.
fn encode_base32(raw: &[u8]) -> String {
    let mut out = String::with_capacity(raw.len() * 8 / 5);
    let mut buffer: u32 = 0;
    let mut bits_in_buffer: u32 = 0;
    for byte in raw {
        buffer = (buffer << 8) | u32::from(*byte);
        bits_in_buffer += 8;
        while bits_in_buffer >= 5 {
            bits_in_buffer -= 5;
            let idx = ((buffer >> bits_in_buffer) & 0b11111) as usize;
            out.push(ALPHABET[idx] as char);
        }
    }
    // 10 bytes * 8 = 80 bits, divisible by 5 → no leftover. (Defensive: if
    // a future caller changes CODE_RAW_BYTES to a non-multiple of 5 we'd
    // pad here, but the assert below makes the invariant explicit.)
    debug_assert_eq!(bits_in_buffer, 0, "10 bytes packs evenly into 16 base32 chars");
    out
}

/// Sizes of the three groups in a formatted recovery code, in order.
/// 5 + 5 + 6 = 16 source characters = 80 bits of entropy.
///
/// The shape mirrors the canonical credit-card-style "5-5-..." grouping
/// called out in the W2 audit spec — three groups, with the trailing
/// group widened to 6 to absorb the 16th character so we don't fall
/// below the 80-bit entropy floor. (A literal 5-5-5 form would drop one
/// character and yield only 75 bits, which the audit explicitly forbids.)
const GROUP_SIZES: [usize; 3] = [5, 5, 6];

/// Format an unformatted 16-character code into three dash-separated
/// groups sized [`GROUP_SIZES`]. Example output:
/// `ABCDE-FGHJK-MNPQRS` (18 chars: 16 payload + 2 dashes).
fn format_groups(code: &str) -> String {
    debug_assert_eq!(
        code.len(),
        GROUP_SIZES.iter().sum::<usize>(),
        "expected 16-char base32 payload"
    );
    let bytes = code.as_bytes();
    let dash_count = GROUP_SIZES.len() - 1;
    let mut out = String::with_capacity(code.len() + dash_count);
    let mut cursor = 0usize;
    for (i, size) in GROUP_SIZES.iter().enumerate() {
        if i > 0 {
            out.push('-');
        }
        out.push_str(std::str::from_utf8(&bytes[cursor..cursor + size]).unwrap());
        cursor += size;
    }
    out
}

/// Normalize user input: strip whitespace and dashes, uppercase.
fn normalize_input(input: &str) -> String {
    input
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-')
        .flat_map(char::to_uppercase)
        .collect()
}

/// Generate a fresh code: 10 random bytes → 16 base32 chars → grouped form.
/// The alphabet is already uppercase Crockford, so no case fold is needed.
fn generate_one() -> String {
    let mut raw = [0u8; CODE_RAW_BYTES];
    rand::rngs::OsRng.fill_bytes(&mut raw);
    let unformatted = encode_base32(&raw);
    format_groups(&unformatted)
}

/// Hash a normalized code with Argon2id. Salt is freshly random per call.
fn hash_code(normalized: &str) -> Result<String, AuthError> {
    let salt = SaltString::generate(&mut PwOsRng);
    code_hasher()
        .hash_password(normalized.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| AuthError::internal(format!("argon2 hash failed (recovery code): {e}")))
}

/// Generate a batch of plaintext codes + their stored Argon2 hashes.
///
/// The plaintext codes are returned to the caller for one-time display to
/// the user; only the hashes ever land in storage.
pub fn generate_batch() -> Result<(Vec<String>, Vec<String>), AuthError> {
    let mut plaintext = Vec::with_capacity(CODES_PER_BATCH);
    let mut hashes = Vec::with_capacity(CODES_PER_BATCH);
    for _ in 0..CODES_PER_BATCH {
        let code = generate_one();
        let normalized = normalize_input(&code);
        let hash = hash_code(&normalized)?;
        plaintext.push(code);
        hashes.push(hash);
    }
    Ok((plaintext, hashes))
}

/// Outcome of a redeem attempt: `Consumed { remaining }` if one matched
/// (and was consumed), `NoMatch` if the supplied code matches none of the
/// stored hashes.
///
/// All stored hashes are scanned even after a match is found so the
/// per-attempt timing does not depend on *which* slot matched. Each
/// Argon2id verification is constant-time internally (subtle), so an
/// attacker probing for matches sees the same "verify ~N hashes" wall time
/// regardless of input.
pub fn redeem(stored_hashes: &[String], candidate: &str) -> Result<RedeemOutcome, AuthError> {
    let normalized = normalize_input(candidate);
    if normalized.is_empty() {
        return Ok(RedeemOutcome::NoMatch);
    }

    let hasher = code_hasher();
    let mut matched_index: Option<usize> = None;

    for (idx, stored) in stored_hashes.iter().enumerate() {
        let parsed = match PasswordHash::new(stored) {
            Ok(p) => p,
            Err(e) => {
                // A corrupt stored hash is a server-side data integrity
                // issue, not a user-facing failure. Log and continue.
                tracing::warn!(idx, error = ?e, "skipping malformed recovery hash slot");
                continue;
            }
        };
        if hasher
            .verify_password(normalized.as_bytes(), &parsed)
            .is_ok()
            && matched_index.is_none()
        {
            matched_index = Some(idx);
            // Don't break — keep scanning to keep timing flat.
        }
    }

    match matched_index {
        Some(idx) => {
            let mut remaining = stored_hashes.to_vec();
            remaining.remove(idx);
            Ok(RedeemOutcome::Consumed { remaining })
        }
        None => Ok(RedeemOutcome::NoMatch),
    }
}

/// Result of a recovery-code redeem attempt.
#[derive(Debug)]
pub enum RedeemOutcome {
    /// The code matched a stored hash; the matching hash was removed.
    /// `remaining` is the post-consume vector the caller should persist.
    Consumed { remaining: Vec<String> },
    /// No stored hash matched.
    NoMatch,
}

/// Async wrapper around [`generate_batch`] that runs the 10-hash batch on
/// the blocking-task pool under the process-global Argon2 semaphore. See
/// the module-level Concurrency note.
pub async fn generate_batch_async() -> Result<(Vec<String>, Vec<String>), AuthError> {
    argon2_pool::run_blocking(generate_batch).await
}

/// Async wrapper around [`redeem`] with the same concurrency semantics as
/// [`generate_batch_async`].
pub async fn redeem_async(
    stored_hashes: &[String],
    candidate: &str,
) -> Result<RedeemOutcome, AuthError> {
    let stored = stored_hashes.to_vec();
    let candidate = candidate.to_owned();
    argon2_pool::run_blocking(move || redeem(&stored, &candidate)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn batch_yields_ten_unique_codes_and_ten_hashes() {
        let (plain, hashes) = generate_batch().expect("batch");
        assert_eq!(plain.len(), CODES_PER_BATCH);
        assert_eq!(hashes.len(), CODES_PER_BATCH);
        let unique: HashSet<_> = plain.iter().collect();
        assert_eq!(unique.len(), CODES_PER_BATCH, "all codes must be unique");
        // Hashes are PHC strings.
        for h in &hashes {
            assert!(h.starts_with("$argon2id$"), "expected PHC string: {h}");
        }
    }

    #[test]
    fn codes_render_in_grouped_form_with_expected_entropy() {
        // Grouped form: 5-5-6 with two dashes → 18 chars total.
        // 16 payload chars × 5 bits = 80 bits of entropy.
        let expected_payload: usize = GROUP_SIZES.iter().sum();
        let expected_dashes = GROUP_SIZES.len() - 1;
        let expected_len = expected_payload + expected_dashes;

        let (plain, _) = generate_batch().unwrap();
        for code in &plain {
            assert_eq!(code.len(), expected_len, "grouped length: {code}");
            assert_eq!(
                code.matches('-').count(),
                expected_dashes,
                "dash count: {code}"
            );
            // Visible characters live in the Crockford alphabet — case
            // sensitive: the alphabet is all uppercase and digits, so any
            // lowercase character coming back from generation is a bug.
            for c in code.chars().filter(|c| *c != '-') {
                assert!(
                    ALPHABET.iter().any(|b| *b as char == c),
                    "char {c} in {code} is outside the Crockford alphabet"
                );
            }
        }
    }

    #[test]
    fn redeem_consumes_a_code_exactly_once() {
        let (plain, hashes) = generate_batch().expect("batch");
        let first = plain[0].clone();
        let outcome = redeem(&hashes, &first).expect("redeem");
        let RedeemOutcome::Consumed { remaining } = outcome else {
            panic!("first redeem must succeed");
        };
        assert_eq!(remaining.len(), hashes.len() - 1);

        // Replay the same code against the post-consume hash set: must fail.
        let replay = redeem(&remaining, &first).expect("replay");
        assert!(matches!(replay, RedeemOutcome::NoMatch));
    }

    #[test]
    fn unknown_code_does_not_match() {
        let (_, hashes) = generate_batch().unwrap();
        // A syntactically valid but unrelated code in the current
        // 5-5-6 grouped form.
        let bogus = "ABCDE-FGHJK-MNPQRS";
        let outcome = redeem(&hashes, bogus).expect("redeem");
        assert!(matches!(outcome, RedeemOutcome::NoMatch));
    }

    #[test]
    fn redeem_is_dash_and_case_insensitive() {
        let (plain, hashes) = generate_batch().unwrap();
        let stripped: String = plain[3].chars().filter(|c| *c != '-').collect();
        let lowered = stripped.to_lowercase();
        let outcome = redeem(&hashes, &lowered).expect("redeem");
        assert!(matches!(outcome, RedeemOutcome::Consumed { .. }));
    }

    #[test]
    fn entropy_floor_is_at_least_eighty_bits() {
        // Each code has 16 base32 chars * 5 bits = 80 bits before grouping
        // is applied. The 5-5-6 grouped shape preserves all 16 source
        // characters; no entropy is dropped at display time.
        let (plain, _) = generate_batch().unwrap();
        let payload_chars: usize = plain[0].chars().filter(|c| *c != '-').count();
        assert!(
            payload_chars * 5 >= 80,
            "each code must carry >= 80 bits of entropy (got {} chars * 5 = {} bits)",
            payload_chars,
            payload_chars * 5
        );
    }

    #[test]
    fn empty_input_is_no_match() {
        let (_, hashes) = generate_batch().unwrap();
        assert!(matches!(
            redeem(&hashes, "").unwrap(),
            RedeemOutcome::NoMatch
        ));
        assert!(matches!(
            redeem(&hashes, "   -   -   ").unwrap(),
            RedeemOutcome::NoMatch
        ));
    }

    // ─────────────────────────────────────────────────────────────────────
    // B-Audit (W2 hardening): Crockford alphabet + grouping + entropy
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn alphabet_is_canonical_crockford() {
        // Digits 0–9 followed by uppercase A–Z minus I, L, O, U.
        assert_eq!(
            std::str::from_utf8(ALPHABET).unwrap(),
            "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
        );
        assert_eq!(ALPHABET.len(), 32, "base32 needs exactly 32 symbols");

        // None of the ambiguity-prone characters can ever appear in the
        // alphabet — the whole point of Crockford's selection.
        for bad in AMBIGUOUS_CHARS {
            assert!(
                !ALPHABET.iter().any(|b| *b as char == *bad),
                "Crockford alphabet must exclude {bad}"
            );
        }
        // And the alphabet is strictly uppercase + digits — no lowercase.
        for b in ALPHABET.iter() {
            assert!(
                b.is_ascii_uppercase() || b.is_ascii_digit(),
                "non-uppercase non-digit byte {:?} in alphabet",
                *b as char
            );
        }
    }

    #[test]
    fn group_sizes_match_spec_and_sum_to_eighty_bits() {
        // Three groups, summing to 16 payload characters → exactly 80
        // bits of entropy.
        assert_eq!(GROUP_SIZES.len(), 3, "three groups");
        assert_eq!(GROUP_SIZES.iter().sum::<usize>(), 16);
        assert_eq!(GROUP_SIZES.iter().sum::<usize>() * 5, 80);
    }

    #[test]
    fn ten_generated_codes_audit_clean() {
        // Audit assertion bundled per the W2 spec:
        //   - 10 codes
        //   - all unique
        //   - all in Crockford alphabet (no I/L/O/U, no lowercase)
        //   - grouping format matches GROUP_SIZES with dashes between
        //   - ≥ 80 bits of entropy per code
        let (plain, _) = generate_batch().unwrap();
        assert_eq!(plain.len(), 10);
        let unique: HashSet<_> = plain.iter().collect();
        assert_eq!(unique.len(), 10, "all ten codes must be unique");

        for code in &plain {
            // No ambiguous chars.
            for bad in AMBIGUOUS_CHARS {
                assert!(
                    !code.contains(*bad),
                    "code {code} contains ambiguous char {bad}"
                );
            }
            // No lowercase letters.
            assert!(
                code.chars().filter(|c| *c != '-').all(|c| !c.is_ascii_lowercase()),
                "code {code} contains lowercase characters"
            );
            // Every payload char is from the canonical alphabet
            // (case-sensitive on purpose).
            for c in code.chars().filter(|c| *c != '-') {
                assert!(
                    ALPHABET.iter().any(|b| *b as char == c),
                    "char {c} in {code} is outside the Crockford alphabet"
                );
            }
            // Grouping check: split on '-', verify each group's size.
            let groups: Vec<&str> = code.split('-').collect();
            assert_eq!(
                groups.len(),
                GROUP_SIZES.len(),
                "expected {} groups, got {} in {code}",
                GROUP_SIZES.len(),
                groups.len()
            );
            for (i, expected) in GROUP_SIZES.iter().enumerate() {
                assert_eq!(
                    groups[i].len(),
                    *expected,
                    "group {i} in {code} should be {expected} chars"
                );
            }
            // Entropy floor.
            let payload_chars: usize = code.chars().filter(|c| *c != '-').count();
            assert!(payload_chars * 5 >= 80, "code {code} has < 80 bits");
        }
    }
}
