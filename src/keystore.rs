//! Encryption-at-rest v2 — envelope keystore (crypto core, phase P1).
//!
//! See `docs/encryption-v2-design.md`. This module is the pure crypto core:
//! envelope key hierarchy (a master key / KEK wrapping per-epoch data keys /
//! DEKs), an Argon2id passphrase unseal path, and the serialisable keystore
//! container with a key fingerprint + crypto/schema version gates.
//!
//! P1 deliberately does NOT wire into storage (that is P2) and ships only the
//! passphrase unseal provider (keychain/KMS/recovery are P5). The structure
//! (multi-wrap `kek_wraps`, epoched `deks`) is in place so later phases add
//! providers and rotation without reshaping the on-disk keystore.
//!
//! Threat model and rationale: `docs/encryption-v2-design.md` §1, §11.

use aes_gcm::aead::{Aead, AeadCore, KeyInit, OsRng, Payload};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use anyhow::{anyhow, Context, Result};
use argon2::{Algorithm, Argon2, Params, Version};
use base64::Engine;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// Envelope wire/format version (records + keystore). Bumped on format changes.
pub const CRYPTO_VERSION: u8 = 1;
/// Storage schema version gate for migrations (see design §11 B-5).
pub const SCHEMA_VERSION: u8 = 1;

const KEY_LEN: usize = 32;
const SALT_LEN: usize = 16;

/// AAD bound into the KEK wrap (domain separation / substitution resistance).
const KEK_AAD: &[u8] = b"veld:keystore:kek:v1";

/// A 256-bit secret (KEK or DEK), memory-zeroized on drop. Never serialised —
/// only its *wrapped* form is persisted.
pub type SecretKey = Zeroizing<[u8; KEY_LEN]>;

fn b64e(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn b64d(s: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .context("invalid base64 in keystore")
}

fn random_key() -> SecretKey {
    let mut k = Zeroizing::new([0u8; KEY_LEN]);
    OsRng.fill_bytes(k.as_mut_slice());
    k
}

fn random_salt() -> [u8; SALT_LEN] {
    let mut s = [0u8; SALT_LEN];
    OsRng.fill_bytes(&mut s);
    s
}

/// Non-secret 4-byte fingerprint of a key: `SHA-256(key)[..4]`. A loss/mismatch
/// tripwire, not an authenticator (see design §6 / §11 B-5).
fn fingerprint(key: &[u8]) -> [u8; 4] {
    let d = Sha256::digest(key);
    [d[0], d[1], d[2], d[3]]
}

/// AAD bound into a DEK wrap — pins the wrap to its epoch so a DEK cannot be
/// silently substituted across epochs.
fn dek_aad(epoch: u32) -> Vec<u8> {
    format!("veld:keystore:dek:v1:epoch:{epoch}").into_bytes()
}

/// AES-256-GCM key-wrap: encrypt `key` under `wrapping_key` with `aad` bound in.
fn wrap_key(wrapping_key: &[u8; KEY_LEN], key: &[u8], aad: &[u8], provider: &str) -> Result<Wrapped> {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(wrapping_key));
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let ciphertext = cipher
        .encrypt(&nonce, Payload { msg: key, aad })
        .map_err(|e| anyhow!("AES-256-GCM key wrap failed: {e}"))?;
    Ok(Wrapped {
        provider: provider.to_string(),
        nonce: b64e(nonce.as_slice()),
        ciphertext: b64e(&ciphertext),
    })
}

/// Inverse of [`wrap_key`]; errors on wrong key/passphrase, tamper, or AAD mismatch.
fn unwrap_key(wrapping_key: &[u8; KEY_LEN], w: &Wrapped, aad: &[u8]) -> Result<SecretKey> {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(wrapping_key));
    let nonce_bytes = b64d(&w.nonce)?;
    if nonce_bytes.len() != 12 {
        return Err(anyhow!("keystore nonce wrong length"));
    }
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = b64d(&w.ciphertext)?;
    let plaintext = cipher
        .decrypt(nonce, Payload { msg: &ciphertext, aad })
        .map_err(|e| anyhow!("key unwrap failed (wrong key/passphrase or tampered keystore): {e}"))?;
    if plaintext.len() != KEY_LEN {
        return Err(anyhow!("unwrapped key has wrong length"));
    }
    let mut k = Zeroizing::new([0u8; KEY_LEN]);
    k.copy_from_slice(&plaintext);
    Ok(k)
}

/// A secret wrapped under some unseal provider (passphrase in P1).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Wrapped {
    /// Unseal provider id: `"passphrase"` (P1); `"keychain"`/`"kms"`/`"recovery"` later.
    pub provider: String,
    /// base64(12-byte GCM nonce).
    pub nonce: String,
    /// base64(ciphertext+tag).
    pub ciphertext: String,
}

/// Argon2id parameters for deriving an unseal key from a passphrase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KdfParams {
    /// Memory cost in KiB.
    pub m_cost: u32,
    /// Iterations.
    pub t_cost: u32,
    /// Parallelism (lanes).
    pub p_cost: u32,
    /// base64(salt).
    pub salt: String,
}

impl KdfParams {
    /// Strong production defaults (~256 MiB, 3 passes). Tune/upgrade over time.
    pub fn production() -> Self {
        Self {
            m_cost: 256 * 1024,
            t_cost: 3,
            p_cost: 1,
            salt: b64e(&random_salt()),
        }
    }

    /// Weak params for fast tests ONLY (never production).
    #[cfg(test)]
    pub fn fast_for_tests() -> Self {
        Self {
            m_cost: 8,
            t_cost: 1,
            p_cost: 1,
            salt: b64e(&random_salt()),
        }
    }

    /// Derive a 256-bit unseal key from the passphrase under these params.
    fn derive(&self, passphrase: &str) -> Result<SecretKey> {
        let params = Params::new(self.m_cost, self.t_cost, self.p_cost, Some(KEY_LEN))
            .map_err(|e| anyhow!("invalid Argon2 params: {e}"))?;
        let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
        let salt = b64d(&self.salt)?;
        let mut out = Zeroizing::new([0u8; KEY_LEN]);
        argon
            .hash_password_into(passphrase.as_bytes(), &salt, out.as_mut_slice())
            .map_err(|e| anyhow!("Argon2id derivation failed: {e}"))?;
        Ok(out)
    }
}

/// A per-epoch data key, wrapped by the master key (KEK).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DekEntry {
    pub epoch: u32,
    /// DEK wrapped by the KEK (provider id `"kek"`).
    pub wrapped: Wrapped,
    /// `"active"` or `"retired"` (rotation, P4).
    pub state: String,
}

/// Serialisable keystore: the master key wrapped per unseal provider, the
/// epoched data keys wrapped by the master key, and version/fingerprint gates.
/// Holds no plaintext key material.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Keystore {
    pub crypto_version: u8,
    pub schema_version: u8,
    pub kdf: KdfParams,
    /// KEK wrapped once per enabled unseal provider (P1: passphrase only).
    pub kek_wraps: Vec<Wrapped>,
    pub deks: Vec<DekEntry>,
    pub active_epoch: u32,
    /// base64(4-byte KEK fingerprint) — detects unseal-to-wrong-key.
    pub kek_fingerprint: String,
}

impl Keystore {
    /// Create a fresh keystore: random KEK + an initial active DEK (epoch 0),
    /// with the KEK wrapped under the passphrase. Returns the keystore only; the
    /// caller persists it and re-unseals to obtain key material.
    pub fn create(passphrase: &str, kdf: KdfParams) -> Result<Self> {
        let kek = random_key();
        let dek = random_key();

        let unseal = kdf.derive(passphrase)?;
        let kek_wrap = wrap_key(&unseal, kek.as_slice(), KEK_AAD, "passphrase")?;
        let dek_wrap = wrap_key(&kek, dek.as_slice(), &dek_aad(0), "kek")?;

        Ok(Self {
            crypto_version: CRYPTO_VERSION,
            schema_version: SCHEMA_VERSION,
            kdf,
            kek_wraps: vec![kek_wrap],
            deks: vec![DekEntry {
                epoch: 0,
                wrapped: dek_wrap,
                state: "active".to_string(),
            }],
            active_epoch: 0,
            kek_fingerprint: b64e(&fingerprint(kek.as_slice())),
        })
    }

    /// Unseal the master key (KEK) via the passphrase provider; verifies the
    /// fingerprint so an unexpected key surfaces as an error, not silent garbage.
    pub fn unseal_with_passphrase(&self, passphrase: &str) -> Result<SecretKey> {
        let unseal = self.kdf.derive(passphrase)?;
        let w = self
            .kek_wraps
            .iter()
            .find(|w| w.provider == "passphrase")
            .ok_or_else(|| anyhow!("keystore has no passphrase wrap"))?;
        let kek = unwrap_key(&unseal, w, KEK_AAD)?;
        if b64e(&fingerprint(kek.as_slice())) != self.kek_fingerprint {
            return Err(anyhow!("keystore KEK fingerprint mismatch after unseal"));
        }
        Ok(kek)
    }

    /// Unwrap the active-epoch data key. Returns `(epoch, dek)`.
    pub fn active_data_key(&self, kek: &SecretKey) -> Result<(u32, SecretKey)> {
        let dek = self.data_key_for_epoch(kek, self.active_epoch)?;
        Ok((self.active_epoch, dek))
    }

    /// Unwrap the data key for a specific epoch (older records may be on prior
    /// epochs; rotation in P4).
    pub fn data_key_for_epoch(&self, kek: &SecretKey, epoch: u32) -> Result<SecretKey> {
        let entry = self
            .deks
            .iter()
            .find(|d| d.epoch == epoch)
            .ok_or_else(|| anyhow!("no data key for epoch {epoch}"))?;
        unwrap_key(kek, &entry.wrapped, &dek_aad(epoch))
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("failed to serialize keystore")
    }

    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("failed to parse keystore")
    }
}

// ============================================================================
// Record envelope (P2) — encrypt/decrypt a serialized record blob under a DEK.
// Wire format: `ENC\0 | crypto_version(1) | epoch(4 LE) | nonce(12) | ct+tag`.
// AAD binds the epoch (prevents cross-epoch substitution). Record-id binding is
// a documented follow-up: it requires confirming the RocksDB key == memory id
// invariant at every write site first (design §5).
// ============================================================================

/// 4-byte marker distinguishing an encrypted record from a legacy plaintext
/// (bincode) record. bincode of a `Memory` never begins with `ENC\0`.
const RECORD_MARKER: &[u8; 4] = b"ENC\x00";
/// marker(4) + version(1) + epoch(4) + nonce(12)
const RECORD_HEADER_LEN: usize = 4 + 1 + 4 + 12;
/// GCM tag is 16 bytes; a valid ciphertext is at least the tag.
const RECORD_MIN_LEN: usize = RECORD_HEADER_LEN + 16;

fn record_aad(epoch: u32) -> Vec<u8> {
    format!("veld:record:v1:epoch:{epoch}").into_bytes()
}

/// `true` if `data` carries the encrypted-record marker.
pub fn is_encrypted_record(data: &[u8]) -> bool {
    data.len() >= RECORD_MARKER.len() && &data[..4] == RECORD_MARKER
}

/// Read the epoch from an encrypted record header, so the reader can select the
/// right DEK before decrypting. Returns `None` if not an encrypted record.
pub fn record_epoch(data: &[u8]) -> Option<u32> {
    if !is_encrypted_record(data) || data.len() < RECORD_HEADER_LEN {
        return None;
    }
    Some(u32::from_le_bytes([data[5], data[6], data[7], data[8]]))
}

/// Decrypt an encrypted record blob with the DEK for its epoch. The caller is
/// responsible for selecting `dek` matching [`record_epoch`].
pub fn decrypt_record(dek: &SecretKey, data: &[u8]) -> Result<Vec<u8>> {
    if !is_encrypted_record(data) {
        return Err(anyhow!("not an encrypted record (missing marker)"));
    }
    if data.len() < RECORD_MIN_LEN {
        return Err(anyhow!(
            "encrypted record too short: {} bytes (min {RECORD_MIN_LEN})",
            data.len()
        ));
    }
    let version = data[4];
    if version != CRYPTO_VERSION {
        return Err(anyhow!("unsupported record crypto version {version}"));
    }
    let epoch = u32::from_le_bytes([data[5], data[6], data[7], data[8]]);
    let nonce = Nonce::from_slice(&data[9..RECORD_HEADER_LEN]);
    let ciphertext = &data[RECORD_HEADER_LEN..];
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(dek.as_slice()));
    cipher
        .decrypt(nonce, Payload { msg: ciphertext, aad: &record_aad(epoch) })
        .map_err(|e| anyhow!("record decrypt failed (wrong key/epoch or tampered): {e}"))
}

/// Encrypts serialized record blobs under a fixed active DEK/epoch. Held by the
/// storage layer (P2 wiring). Reads of older epochs go through [`decrypt_record`]
/// with the DEK fetched from the keystore for that epoch.
pub struct RecordCryptor {
    epoch: u32,
    dek: SecretKey,
}

impl RecordCryptor {
    pub fn new(epoch: u32, dek: SecretKey) -> Self {
        Self { epoch, dek }
    }

    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Encrypt a serialized record blob under the active DEK.
    pub fn encrypt_record(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(self.dek.as_slice()));
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(&nonce, Payload { msg: plaintext, aad: &record_aad(self.epoch) })
            .map_err(|e| anyhow!("record encrypt failed: {e}"))?;
        let mut out = Vec::with_capacity(RECORD_HEADER_LEN + ciphertext.len());
        out.extend_from_slice(RECORD_MARKER);
        out.push(CRYPTO_VERSION);
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(nonce.as_slice());
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt with the active DEK (convenience for same-epoch reads/tests).
    pub fn decrypt_record(&self, data: &[u8]) -> Result<Vec<u8>> {
        decrypt_record(&self.dek, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_unseal_roundtrip() {
        let ks = Keystore::create("correct horse battery staple", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("correct horse battery staple").unwrap();
        let (epoch, dek) = ks.active_data_key(&kek).unwrap();
        assert_eq!(epoch, 0);
        assert_eq!(dek.len(), KEY_LEN);
    }

    #[test]
    fn wrong_passphrase_fails() {
        let ks = Keystore::create("right-passphrase", KdfParams::fast_for_tests()).unwrap();
        assert!(ks.unseal_with_passphrase("wrong-passphrase").is_err());
    }

    #[test]
    fn json_roundtrip_preserves_unseal() {
        let ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let json = ks.to_json().unwrap();
        let ks2 = Keystore::from_json(&json).unwrap();
        let kek = ks2.unseal_with_passphrase("pw").unwrap();
        ks2.active_data_key(&kek).unwrap();
        // The raw KEK must never appear in the serialised keystore (only wrapped).
        assert!(
            !json.contains(&b64e(kek.as_slice())),
            "raw KEK must not appear in serialized keystore"
        );
        assert_eq!(ks.kek_fingerprint, ks2.kek_fingerprint);
        assert_eq!(ks2.crypto_version, CRYPTO_VERSION);
        assert_eq!(ks2.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn argon2_is_deterministic_for_same_params() {
        let kdf = KdfParams::fast_for_tests();
        let a = kdf.derive("pw").unwrap();
        let b = kdf.derive("pw").unwrap();
        assert_eq!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn wrap_unwrap_roundtrip_and_aad_binding() {
        let wk = [7u8; KEY_LEN];
        let secret = [9u8; KEY_LEN];
        let w = wrap_key(&wk, &secret, b"aad-a", "x").unwrap();

        // correct key + AAD recovers the secret
        let got = unwrap_key(&wk, &w, b"aad-a").unwrap();
        assert_eq!(got.as_slice(), &secret);

        // wrong AAD fails (binding holds)
        assert!(unwrap_key(&wk, &w, b"aad-b").is_err());
        // wrong wrapping key fails
        assert!(unwrap_key(&[8u8; KEY_LEN], &w, b"aad-a").is_err());
    }

    #[test]
    fn epoch_aad_prevents_cross_epoch_unwrap() {
        // A DEK wrapped for epoch 0 must not unwrap under epoch 1's AAD.
        let kek = [3u8; KEY_LEN];
        let dek = [4u8; KEY_LEN];
        let w = wrap_key(&kek, &dek, &dek_aad(0), "kek").unwrap();
        assert!(unwrap_key(&kek, &w, &dek_aad(1)).is_err());
        assert!(unwrap_key(&kek, &w, &dek_aad(0)).is_ok());
    }

    #[test]
    fn fingerprint_is_stable_and_distinguishing() {
        assert_eq!(fingerprint(&[1u8; KEY_LEN]), fingerprint(&[1u8; KEY_LEN]));
        assert_ne!(fingerprint(&[1u8; KEY_LEN]), fingerprint(&[2u8; KEY_LEN]));
    }

    #[test]
    fn record_encrypt_decrypt_roundtrip() {
        let cryptor = RecordCryptor::new(0, Zeroizing::new([5u8; KEY_LEN]));
        let plaintext = b"bincode-serialized-memory-record-bytes".to_vec();
        let enc = cryptor.encrypt_record(&plaintext).unwrap();
        assert!(is_encrypted_record(&enc));
        assert_eq!(record_epoch(&enc), Some(0));
        assert_ne!(enc, plaintext);
        assert_eq!(cryptor.decrypt_record(&enc).unwrap(), plaintext);
    }

    #[test]
    fn legacy_plaintext_is_not_marked_encrypted() {
        // A typical bincode blob won't start with ENC\0.
        let plain = vec![0x10, 0x00, 0x00, 0x00, 0x42];
        assert!(!is_encrypted_record(&plain));
        assert_eq!(record_epoch(&plain), None);
    }

    #[test]
    fn record_wrong_dek_fails() {
        let c1 = RecordCryptor::new(0, Zeroizing::new([1u8; KEY_LEN]));
        let enc = c1.encrypt_record(b"secret").unwrap();
        assert!(decrypt_record(&Zeroizing::new([2u8; KEY_LEN]), &enc).is_err());
    }

    #[test]
    fn record_tamper_is_detected() {
        let c = RecordCryptor::new(3, Zeroizing::new([7u8; KEY_LEN]));
        let mut enc = c.encrypt_record(b"payload").unwrap();
        let last = enc.len() - 1;
        enc[last] ^= 0xff; // flip a ciphertext byte
        assert!(c.decrypt_record(&enc).is_err());
    }

    #[test]
    fn record_epoch_in_header_matches() {
        let c = RecordCryptor::new(42, Zeroizing::new([9u8; KEY_LEN]));
        let enc = c.encrypt_record(b"x").unwrap();
        assert_eq!(record_epoch(&enc), Some(42));
        assert_eq!(c.epoch(), 42);
    }

    #[test]
    fn record_from_keystore_dek_end_to_end() {
        let ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        let (epoch, dek) = ks.active_data_key(&kek).unwrap();
        let cryptor = RecordCryptor::new(epoch, dek);
        let enc = cryptor.encrypt_record(b"end-to-end").unwrap();
        assert_eq!(cryptor.decrypt_record(&enc).unwrap(), b"end-to-end");
    }
}
