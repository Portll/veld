//! Read side of the retired field-level encryption, kept only to migrate it.
//!
//! Before the record envelope (`keystore`), veld v0.7.8 to v0.7.18 encrypted
//! `experience.content` alone: base64 of `ENC\0 | 12-byte nonce | AES-256-GCM
//! ciphertext+tag`, with no associated data, under the raw key in
//! `VELD_ENCRYPTION_KEY`. Nothing writes that format any more. A record still
//! carrying it is opened here and flagged for migration, so the read that finds it
//! rewrites the whole record sealed under the keystore. A record the earlier record-level migration sealed without opening
//! its content still carries the old ciphertext inside the envelope; it is opened
//! the same way after the envelope comes off.
//!
//! Fail closed, both ways:
//! - legacy content with no key, or content that will not decrypt, is an error, never
//!   ciphertext handed back as the memory's text;
//! - the key is accepted only beside an active keystore. Alone it would mean legacy
//!   records read decrypted and every write after them stored in plaintext.

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, Key, KeyInit, Nonce};
use anyhow::{anyhow, Context, Result};
use base64::Engine;
use zeroize::Zeroizing;

pub const LEGACY_KEY_ENV: &str = "VELD_ENCRYPTION_KEY";

const MARKER: &[u8; 4] = b"ENC\x00";
const NONCE_LEN: usize = 12;
const TAG_LEN: usize = 16;

/// The legacy key, read at call time; `None` when unset or empty.
pub fn legacy_key() -> Result<Option<Zeroizing<[u8; 32]>>> {
    match std::env::var(LEGACY_KEY_ENV) {
        Ok(raw) => parse_legacy_key(&Zeroizing::new(raw)),
        Err(_) => Ok(None),
    }
}

/// Parse a legacy key value: 32 bytes as 64 hex or base64 characters. Empty is
/// `None`; anything else malformed is an error, never treated as unset.
pub fn parse_legacy_key(raw: &str) -> Result<Option<Zeroizing<[u8; 32]>>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let decoded = if trimmed.len() == 64 && trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
        hex::decode(trimmed).ok()
    } else {
        base64::engine::general_purpose::STANDARD
            .decode(trimmed)
            .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(trimmed))
            .ok()
    };
    match decoded {
        Some(bytes) if bytes.len() == 32 => {
            let bytes = Zeroizing::new(bytes);
            let mut key = Zeroizing::new([0u8; 32]);
            key.copy_from_slice(&bytes);
            Ok(Some(key))
        }
        _ => Err(anyhow!(
            "{LEGACY_KEY_ENV} must be 32 bytes as 64 hex or base64 characters; got {} characters",
            trimmed.len()
        )),
    }
}

/// Refuse a legacy key that has no keystore to migrate into. Called once the
/// storage crypto has been set up.
pub fn check_config(keystore_active: bool) -> Result<()> {
    check_config_with(legacy_key()?.is_some(), keystore_active)
}

fn check_config_with(legacy_key_set: bool, keystore_active: bool) -> Result<()> {
    if !legacy_key_set {
        return Ok(());
    }
    if !keystore_active {
        return Err(anyhow!(
            "{LEGACY_KEY_ENV} is set but no keystore is active. Field-level encryption is retired; \
             set VELD_MASTER_PASSPHRASE so a keystore is created, and records in the old format \
             are re-sealed under it as they are read. Starting without one would store every \
             later write in plaintext"
        ));
    }
    tracing::warn!(
        "{LEGACY_KEY_ENV} is set: records in the retired field-level format are decrypted and \
         re-sealed under the keystore as they are read. Unset it once the store has been migrated"
    );
    Ok(())
}

fn legacy_payload(content: &str) -> Option<Vec<u8>> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(content)
        .ok()?;
    (bytes.len() >= MARKER.len() + NONCE_LEN + TAG_LEN && bytes.starts_with(MARKER))
        .then_some(bytes)
}

/// The plaintext of `content` when it is in the retired format, `None` when it is
/// ordinary text.
pub fn open_legacy_content(content: &str) -> Result<Option<String>> {
    if legacy_payload(content).is_none() {
        return Ok(None);
    }
    open_legacy_content_with(content, legacy_key()?.as_deref())
}

fn open_legacy_content_with(content: &str, key: Option<&[u8; 32]>) -> Result<Option<String>> {
    let Some(data) = legacy_payload(content) else {
        return Ok(None);
    };
    let key = key.ok_or_else(|| {
        anyhow!(
            "a memory's content is in the retired field-level encryption format and \
             {LEGACY_KEY_ENV} is unset; set it beside the keystore to migrate the record"
        )
    })?;
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    let nonce = Nonce::from_slice(&data[MARKER.len()..MARKER.len() + NONCE_LEN]);
    let plain = cipher
        .decrypt(nonce, &data[MARKER.len() + NONCE_LEN..])
        .map_err(|_| {
            anyhow!(
                "a memory's legacy field-encrypted content did not decrypt under \
                 {LEGACY_KEY_ENV} (wrong key or corrupted record)"
            )
        })?;
    String::from_utf8(plain)
        .map(Some)
        .context("legacy field-encrypted content decrypted to invalid UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_gcm::aead::OsRng;
    use aes_gcm::AeadCore;

    fn seal(key: &[u8; 32], plain: &str) -> String {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ct = cipher.encrypt(&nonce, plain.as_bytes()).unwrap();
        let mut out = MARKER.to_vec();
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ct);
        base64::engine::general_purpose::STANDARD.encode(out)
    }

    #[test]
    fn ordinary_text_is_left_alone_even_when_it_is_base64() {
        assert_eq!(open_legacy_content("plain words").unwrap(), None);
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"not a legacy payload at all");
        assert_eq!(open_legacy_content(&b64).unwrap(), None);
    }

    #[test]
    fn the_legacy_format_opens_with_its_key_and_fails_closed_without_it() {
        let key = [7u8; 32];
        let sealed = seal(&key, "a remembered secret");
        assert!(
            open_legacy_content_with(&sealed, None).is_err(),
            "no key: an error, not ciphertext as text"
        );
        assert_eq!(
            open_legacy_content_with(&sealed, Some(&key))
                .unwrap()
                .as_deref(),
            Some("a remembered secret")
        );
        assert!(
            open_legacy_content_with(&sealed, Some(&[8u8; 32])).is_err(),
            "wrong key: an error"
        );
    }

    #[test]
    fn a_malformed_key_is_refused_never_treated_as_unset() {
        assert!(parse_legacy_key("short").is_err());
        assert!(parse_legacy_key("  ").unwrap().is_none());
        assert_eq!(
            *parse_legacy_key(&hex::encode([7u8; 32])).unwrap().unwrap(),
            [7u8; 32]
        );
    }

    #[test]
    fn a_legacy_key_is_accepted_only_beside_a_keystore() {
        assert!(check_config_with(true, false).is_err());
        assert!(check_config_with(true, true).is_ok());
        assert!(check_config_with(false, false).is_ok());
    }
}
