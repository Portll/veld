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
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use hmac::{Hmac, Mac};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use subtle::ConstantTimeEq;
use zeroize::{Zeroize, Zeroizing};

/// Envelope wire/format version (records + keystore). Bumped on format changes.
pub const CRYPTO_VERSION: u8 = 1;
/// Storage schema version gate for migrations (see design §11 B-5).
pub const SCHEMA_VERSION: u8 = 1;

const KEY_LEN: usize = 32;
const SALT_LEN: usize = 16;
const NONCE_SIZE: usize = 12;

/// Upper bounds on Argon2 params read from a (possibly tampered) keystore.json,
/// so a malicious keystore cannot trigger a multi-TB allocation / OOM at unseal
/// (the DoS happens before any key check). The 4 GiB ceiling is far above a sane
/// production setting (~256 MiB) yet blocks the absurd.
const MAX_ARGON2_M_COST_KIB: u32 = 4 * 1024 * 1024;
const MAX_ARGON2_T_COST: u32 = 64;
const MAX_ARGON2_P_COST: u32 = 64;

/// Lower bounds on Argon2 params read from a (possibly tampered) keystore.json.
/// The unseal path runs the KDF to derive the wrap key BEFORE the KEK-keyed
/// integrity MAC can be checked, so a tampered keystore could otherwise pin the
/// absolute-weakest params (e.g. m_cost=8 KiB) and make the passphrase wrap
/// cheaply brute-forced offline. 8 MiB is a defense-in-depth floor — well below
/// the ~256 MiB production setting, but far above the absurd downgrade.
const MIN_ARGON2_M_COST_KIB: u32 = 8 * 1024;
const MIN_ARGON2_T_COST: u32 = 1;
const MIN_ARGON2_P_COST: u32 = 1;

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

/// Derive the recovery-code wrapping key. The recovery code is high-entropy, so a
/// fast hash (not Argon2) suffices — there is nothing to brute-force. Grouping
/// dashes, spaces and case are ignored, so a code typed back from a kit opens.
fn recovery_key_from_code(code: &str) -> [u8; KEY_LEN] {
    let digest = Sha256::digest(normalise_code(code).as_bytes());
    let mut k = [0u8; KEY_LEN];
    k.copy_from_slice(&digest);
    k
}

/// Generate a high-entropy, transcribable recovery code (192 bits → 48 hex chars).
pub fn generate_recovery_code() -> String {
    let mut raw = [0u8; 24];
    OsRng.fill_bytes(&mut raw);
    hex::encode(raw)
}

pub const PROVIDER_PASSPHRASE: &str = "passphrase";
pub const PROVIDER_RECOVERY: &str = "recovery";
pub const PROVIDER_KMS: &str = "kms";
/// A tenant keystore's master key sealed under its keyring (see `crate::keyring`).
pub const PROVIDER_RING: &str = "ring";
pub const PROVIDER_PASSKEY: &str = "passkey";
pub const PROVIDER_SECURE_ENCLAVE: &str = "secure-enclave";

/// AAD bound into the root wrap written by the first master-key rotation.
const ROOT_AAD: &[u8] = b"veld:keystore:root:v1";

/// Something that unwraps a keystore's master key.
pub enum Factor<'a> {
    Passphrase(&'a str),
    RecoveryCode(&'a str),
    Kms(&'a dyn KmsClient),
    /// A secret released by something the operator holds or the keyring lends: a
    /// passkey's PRF output, a Secure Enclave key's secret, the ring key. At least
    /// 32 bytes of key material.
    Secret {
        provider: &'a str,
        id: &'a str,
        secret: &'a [u8],
    },
}

impl Factor<'_> {
    fn provider(&self) -> &str {
        match self {
            Factor::Passphrase(_) => PROVIDER_PASSPHRASE,
            Factor::RecoveryCode(_) => PROVIDER_RECOVERY,
            Factor::Kms(_) => PROVIDER_KMS,
            Factor::Secret { provider, .. } => provider,
        }
    }
}

/// A wrap of the master key, described without anything secret.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FactorInfo {
    pub provider: String,
    pub id: String,
    pub label: Option<String>,
    pub created_at: Option<String>,
}

fn wrap_id(w: &Wrapped) -> &str {
    w.id.as_deref().unwrap_or(&w.provider)
}

fn factor_info(w: &Wrapped) -> FactorInfo {
    FactorInfo {
        provider: w.provider.clone(),
        id: wrap_id(w).to_string(),
        label: w.label.clone(),
        created_at: w.created_at.clone(),
    }
}

/// HKDF-SHA256 (RFC 5869) with a single output block.
fn hkdf_sha256(salt: &[u8], ikm: &[u8], info: &[u8]) -> SecretKey {
    let mut extract =
        <HmacSha256 as Mac>::new_from_slice(salt).expect("HMAC accepts any key length");
    Mac::update(&mut extract, ikm);
    let mut prk = Zeroizing::new([0u8; KEY_LEN]);
    prk.copy_from_slice(&Mac::finalize(extract).into_bytes());
    let mut expand =
        <HmacSha256 as Mac>::new_from_slice(prk.as_slice()).expect("HMAC accepts any key length");
    Mac::update(&mut expand, info);
    Mac::update(&mut expand, &[1]);
    let okm = Mac::finalize(expand).into_bytes();
    let mut key = Zeroizing::new([0u8; KEY_LEN]);
    key.copy_from_slice(&okm[..KEY_LEN]);
    key
}

/// The key a device secret wraps the master key under, bound to its provider so one
/// device's secret cannot stand in for another kind.
fn secret_wrap_key(provider: &str, secret: &[u8]) -> Result<SecretKey> {
    if secret.len() < KEY_LEN {
        return Err(anyhow!(
            "a {provider} secret must carry at least {KEY_LEN} bytes, got {}",
            secret.len()
        ));
    }
    Ok(hkdf_sha256(
        b"veld:keystore:factor:v1",
        secret,
        provider.as_bytes(),
    ))
}

/// Stable id of a recovery code: one code enrolled in several keystores carries the
/// same id in each. 32 bits of a hash of a 192-bit code says nothing usable about it.
fn recovery_code_id(code: &str) -> String {
    let digest = Sha256::digest(
        [
            b"veld:recovery-id:".as_slice(),
            normalise_code(code).as_bytes(),
        ]
        .concat(),
    );
    format!("recovery-{}", hex::encode(&digest[..4]))
}

fn normalise_code(code: &str) -> String {
    code.chars()
        .filter(|c| !c.is_whitespace() && *c != '-')
        .collect::<String>()
        .to_ascii_lowercase()
}

/// AES-256-GCM key-wrap: encrypt `key` under `wrapping_key` with `aad` bound in.
fn wrap_key(
    wrapping_key: &[u8; KEY_LEN],
    key: &[u8],
    aad: &[u8],
    provider: &str,
) -> Result<Wrapped> {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(wrapping_key));
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let ciphertext = cipher
        .encrypt(&nonce, Payload { msg: key, aad })
        .map_err(|e| anyhow!("AES-256-GCM key wrap failed: {e}"))?;
    Ok(Wrapped {
        provider: provider.to_string(),
        id: None,
        label: None,
        created_at: None,
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
    let mut plaintext = cipher
        .decrypt(
            nonce,
            Payload {
                msg: &ciphertext,
                aad,
            },
        )
        .map_err(|e| {
            anyhow!("key unwrap failed (wrong key/passphrase or tampered keystore): {e}")
        })?;
    if plaintext.len() != KEY_LEN {
        return Err(anyhow!("unwrapped key has wrong length"));
    }
    let mut k = Zeroizing::new([0u8; KEY_LEN]);
    k.copy_from_slice(&plaintext);
    plaintext.zeroize(); // wipe the intermediate key-material buffer
    Ok(k)
}

/// Decode a 32-byte key from 64-char hex or 44-char base64.
fn decode_32(s: &str) -> Result<[u8; KEY_LEN]> {
    let t = s.trim();
    let bytes = if t.len() == 64 && t.bytes().all(|b| b.is_ascii_hexdigit()) {
        hex::decode(t).context("invalid hex key")?
    } else {
        b64d(t)?
    };
    if bytes.len() != KEY_LEN {
        return Err(anyhow!(
            "key must decode to {KEY_LEN} bytes, got {}",
            bytes.len()
        ));
    }
    let mut k = [0u8; KEY_LEN];
    k.copy_from_slice(&bytes);
    Ok(k)
}

/// A key-management service that wraps/unwraps the master key (P5). Real cloud
/// KMS (AWS/GCP/Vault) implements this with server-side Encrypt/Decrypt calls;
/// the master key never leaves wrapped form on this host. `LocalAeadKms` is the
/// self-hosted concrete impl; an OS-keychain provider implements it too.
pub trait KmsClient {
    fn kms_encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>>;
    fn kms_decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>>;
}

/// Self-hosted KMS: AES-256-GCM under a wrapping key sourced from
/// `VELD_KMS_WRAP_KEY` (64-hex / 44-base64). Ciphertext is `[nonce][ct+tag]`.
/// Lets a server unseal unattended without a passphrase prompt while keeping the
/// wrapping key off the data disk (env/secret-store). Swap for a cloud KMS by
/// implementing `KmsClient`.
pub struct LocalAeadKms {
    key: SecretKey,
}

impl LocalAeadKms {
    pub fn new(key: [u8; KEY_LEN]) -> Self {
        Self {
            key: Zeroizing::new(key),
        }
    }

    /// Load from `VELD_KMS_WRAP_KEY`; `Ok(None)` if unset/empty.
    pub fn from_env() -> Result<Option<Self>> {
        match std::env::var("VELD_KMS_WRAP_KEY") {
            Ok(s) if !s.trim().is_empty() => Ok(Some(Self::new(decode_32(&s)?))),
            _ => Ok(None),
        }
    }
}

impl KmsClient for LocalAeadKms {
    fn kms_encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(self.key.as_slice()));
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ct = cipher
            .encrypt(&nonce, plaintext)
            .map_err(|e| anyhow!("KMS encrypt failed: {e}"))?;
        let mut out = Vec::with_capacity(NONCE_SIZE + ct.len());
        out.extend_from_slice(nonce.as_slice());
        out.extend_from_slice(&ct);
        Ok(out)
    }

    fn kms_decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>> {
        if ciphertext.len() < NONCE_SIZE + 16 {
            return Err(anyhow!("KMS ciphertext too short"));
        }
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(self.key.as_slice()));
        let nonce = Nonce::from_slice(&ciphertext[..NONCE_SIZE]);
        cipher
            .decrypt(nonce, &ciphertext[NONCE_SIZE..])
            .map_err(|e| anyhow!("KMS decrypt failed (wrong KMS key or tampered): {e}"))
    }
}

/// A secret wrapped under some unseal provider (passphrase in P1).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Wrapped {
    /// Unseal provider id: `"passphrase"` (P1); `"keychain"`/`"kms"`/`"recovery"` later.
    pub provider: String,
    /// Tells apart wraps of one provider: each recovery code, passkey or Secure
    /// Enclave key. Absent on wraps written when a provider could have only one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
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

    /// Reduced params for fast tests ONLY (never production). Sits at the
    /// minimum accepted floor so tests still exercise the real KDF cheaply.
    #[cfg(test)]
    pub fn fast_for_tests() -> Self {
        Self {
            m_cost: MIN_ARGON2_M_COST_KIB,
            t_cost: 1,
            p_cost: 1,
            salt: b64e(&random_salt()),
        }
    }

    /// Derive a 256-bit unseal key from the passphrase under these params.
    fn derive(&self, passphrase: &str) -> Result<SecretKey> {
        if self.m_cost > MAX_ARGON2_M_COST_KIB
            || self.t_cost > MAX_ARGON2_T_COST
            || self.p_cost > MAX_ARGON2_P_COST
        {
            return Err(anyhow!(
                "Argon2 params exceed safe bounds (possible tampered keystore): \
                 m_cost={} t_cost={} p_cost={}",
                self.m_cost,
                self.t_cost,
                self.p_cost
            ));
        }
        if self.m_cost < MIN_ARGON2_M_COST_KIB
            || self.t_cost < MIN_ARGON2_T_COST
            || self.p_cost < MIN_ARGON2_P_COST
        {
            return Err(anyhow!(
                "Argon2 params below safe minimum (possible tampered keystore / KDF downgrade): \
                 m_cost={} t_cost={} p_cost={}",
                self.m_cost,
                self.t_cost,
                self.p_cost
            ));
        }
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
    /// The derivation root wrapped under the KEK, written by the first master-key
    /// rotation. Purpose keys and the index blinder derive from the root, so a
    /// rotation leaves them, and everything sealed or blinded under them, unchanged.
    /// Absent until then: the root is the KEK itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_wrap: Option<Wrapped>,
    /// Monotonic generation, bumped on every keystore mutation. Combined with a
    /// database-side sentinel to detect rollback to an older keystore file.
    #[serde(default)]
    pub generation: u64,
    /// HMAC over the keystore (keyed by a KEK-derived integrity key) detecting
    /// in-place tampering of active_epoch / deks / wraps. Empty until sealed.
    #[serde(default)]
    pub mac: String,
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

        let mut ks = Self {
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
            root_wrap: None,
            generation: 0,
            mac: String::new(),
        };
        ks.seal(&kek)?;
        Ok(ks)
    }

    /// Recompute and store the integrity MAC. Call after every mutation while the
    /// KEK is in hand.
    pub fn seal(&mut self, kek: &SecretKey) -> Result<()> {
        self.mac = self.compute_mac(kek)?;
        Ok(())
    }

    /// Verify the keystore has not been tampered in place (active_epoch, deks, or
    /// wraps). Requires the unsealed KEK.
    pub fn verify_integrity(&self, kek: &SecretKey) -> Result<()> {
        if self.mac.is_empty() {
            return Err(anyhow!("keystore is not integrity-sealed (missing MAC)"));
        }
        let expected = self.compute_mac(kek)?;
        let ok = expected.len() == self.mac.len()
            && bool::from(expected.as_bytes().ct_eq(self.mac.as_bytes()));
        if ok {
            Ok(())
        } else {
            Err(anyhow!(
                "keystore integrity check failed — tampered active_epoch/deks/wraps?"
            ))
        }
    }

    /// HMAC-SHA256 over the keystore JSON (with the `mac` field blanked), keyed by
    /// a KEK-derived integrity key.
    fn compute_mac(&self, kek: &SecretKey) -> Result<String> {
        let mut bare = self.clone();
        bare.mac = String::new();
        let bytes = serde_json::to_string(&bare).context("serialize keystore for MAC")?;

        let mut ik = <HmacSha256 as Mac>::new_from_slice(kek.as_slice())
            .expect("HMAC accepts any key length");
        Mac::update(&mut ik, b"veld:keystore-integrity:v1");
        let ikey = Mac::finalize(ik).into_bytes();

        let mut mac =
            <HmacSha256 as Mac>::new_from_slice(&ikey).expect("HMAC accepts any key length");
        Mac::update(&mut mac, bytes.as_bytes());
        Ok(hex::encode(Mac::finalize(mac).into_bytes()))
    }

    /// Unseal the master key (KEK) via the passphrase provider; verifies the
    /// fingerprint so an unexpected key surfaces as an error, not silent garbage.
    pub fn unseal_with_passphrase(&self, passphrase: &str) -> Result<SecretKey> {
        self.unseal(&Factor::Passphrase(passphrase))
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

    /// Rotate the passphrase: re-derive a fresh unseal key (new salt/params) and
    /// re-wrap the SAME master key under it. O(1) — records and indexes are
    /// untouched (they are keyed by the DEK, not the passphrase). Other wraps
    /// (e.g. recovery) are preserved.
    pub fn rotate_passphrase(&mut self, old: &str, new: &str) -> Result<()> {
        let kek = self.unseal_with_passphrase(old)?;
        self.set_passphrase(&kek, new)
    }

    /// (Re)wrap the master key under a NEW passphrase, given an already-unsealed
    /// KEK. This is the recovery primitive: when the old passphrase is lost,
    /// unseal via the recovery code (or KMS) to obtain the KEK, then install a
    /// usable passphrase. Derives a fresh salt/params and replaces only the
    /// passphrase wrap; the recovery and KMS wraps are preserved.
    pub fn set_passphrase(&mut self, kek: &SecretKey, new: &str) -> Result<()> {
        let new_kdf = KdfParams::production();
        let unseal = new_kdf.derive(new)?;
        let new_wrap = wrap_key(&unseal, kek.as_slice(), KEK_AAD, "passphrase")?;
        self.kek_wraps.retain(|w| w.provider != "passphrase");
        self.kek_wraps.push(new_wrap);
        self.kdf = new_kdf;
        self.generation += 1;
        self.seal(kek)?;
        Ok(())
    }

    /// Add (or replace) a recovery wrap of the master key and return the one-time
    /// recovery code. Display it ONCE and never persist it — only the wrap is
    /// stored. Survives passphrase loss as long as the keystore file survives.
    pub fn add_recovery_code(&mut self, kek: &SecretKey) -> Result<String> {
        let code = generate_recovery_code();
        let wrap_key_bytes = recovery_key_from_code(&code);
        let wrap = wrap_key(&wrap_key_bytes, kek.as_slice(), KEK_AAD, "recovery")?;
        self.kek_wraps.retain(|w| w.provider != "recovery");
        self.kek_wraps.push(wrap);
        self.generation += 1;
        self.seal(kek)?;
        Ok(code)
    }

    /// Unseal the master key via any of its recovery codes.
    pub fn unseal_with_recovery_code(&self, code: &str) -> Result<SecretKey> {
        self.unseal(&Factor::RecoveryCode(code))
    }

    /// Rotate the active data key: generate a new epoch's DEK, retire the previous
    /// active one (kept for reads of old records), and make the new one active.
    /// Returns the new epoch. Records re-encrypt lazily on next write or via an
    /// offline pass; old epochs stay readable via [`data_key_for_epoch`].
    pub fn rotate_dek(&mut self, kek: &SecretKey) -> Result<u32> {
        let new_epoch = self.deks.iter().map(|d| d.epoch).max().unwrap_or(0) + 1;
        let dek = random_key();
        let wrap = wrap_key(kek, dek.as_slice(), &dek_aad(new_epoch), "kek")?;
        for d in &mut self.deks {
            if d.state == "active" {
                d.state = "retired".to_string();
            }
        }
        self.deks.push(DekEntry {
            epoch: new_epoch,
            wrapped: wrap,
            state: "active".to_string(),
        });
        self.active_epoch = new_epoch;
        self.generation += 1;
        self.seal(kek)?;
        Ok(new_epoch)
    }

    /// Add (or replace) a KMS wrap of the master key, enabling unattended unseal
    /// without a passphrase. The KMS holds the wrapping key; only the wrapped KEK
    /// is stored here.
    pub fn add_kms_wrap(&mut self, kek: &SecretKey, kms: &dyn KmsClient) -> Result<()> {
        let blob = kms.kms_encrypt(kek.as_slice())?;
        let wrap = Wrapped {
            provider: PROVIDER_KMS.to_string(),
            id: None,
            label: None,
            created_at: None,
            nonce: String::new(), // KMS ciphertext is self-framed
            ciphertext: b64e(&blob),
        };
        self.kek_wraps.retain(|w| w.provider != "kms");
        self.kek_wraps.push(wrap);
        self.generation += 1;
        self.seal(kek)?;
        Ok(())
    }

    /// Unseal the master key via the KMS provider.
    pub fn unseal_with_kms(&self, kms: &dyn KmsClient) -> Result<SecretKey> {
        self.unseal(&Factor::Kms(kms))
    }

    /// Every wrap of the master key, described without anything secret.
    pub fn factors(&self) -> Vec<FactorInfo> {
        self.kek_wraps.iter().map(factor_info).collect()
    }

    /// The master key, unwrapped by `factor` and checked against the fingerprint.
    pub fn unseal(&self, factor: &Factor) -> Result<SecretKey> {
        Ok(self.open_wrap(factor)?.1)
    }

    /// The wrap `factor` opens.
    pub fn identify(&self, factor: &Factor) -> Result<FactorInfo> {
        let (index, _) = self.open_wrap(factor)?;
        Ok(factor_info(&self.kek_wraps[index]))
    }

    /// Whether this keystore carries the ring wrap `ring_id`.
    pub fn has_ring(&self, ring_id: &str) -> bool {
        self.kek_wraps
            .iter()
            .any(|w| w.provider == PROVIDER_RING && wrap_id(w) == ring_id)
    }

    fn open_wrap(&self, factor: &Factor) -> Result<(usize, SecretKey)> {
        let (index, kek) = match factor {
            Factor::Passphrase(passphrase) => {
                let index = self
                    .kek_wraps
                    .iter()
                    .position(|w| w.provider == PROVIDER_PASSPHRASE)
                    .ok_or_else(|| anyhow!("keystore has no passphrase wrap"))?;
                let unseal = self.kdf.derive(passphrase)?;
                (index, unwrap_key(&unseal, &self.kek_wraps[index], KEK_AAD)?)
            }
            Factor::RecoveryCode(code) => {
                let key = recovery_key_from_code(code);
                self.kek_wraps
                    .iter()
                    .enumerate()
                    .filter(|(_, w)| w.provider == PROVIDER_RECOVERY)
                    .find_map(|(i, w)| unwrap_key(&key, w, KEK_AAD).ok().map(|k| (i, k)))
                    .ok_or_else(|| anyhow!("the recovery code opens no wrap in this keystore"))?
            }
            Factor::Kms(kms) => {
                let index = self
                    .kek_wraps
                    .iter()
                    .position(|w| w.provider == PROVIDER_KMS)
                    .ok_or_else(|| anyhow!("keystore has no KMS wrap"))?;
                let mut pt = kms.kms_decrypt(&b64d(&self.kek_wraps[index].ciphertext)?)?;
                if pt.len() != KEY_LEN {
                    pt.zeroize();
                    return Err(anyhow!("KMS-unwrapped key has wrong length"));
                }
                let mut kek = Zeroizing::new([0u8; KEY_LEN]);
                kek.copy_from_slice(&pt);
                pt.zeroize(); // wipe the intermediate key-material buffer
                (index, kek)
            }
            Factor::Secret {
                provider,
                id,
                secret,
            } => {
                let index = self
                    .kek_wraps
                    .iter()
                    .position(|w| w.provider == *provider && wrap_id(w) == *id)
                    .ok_or_else(|| anyhow!("keystore has no {provider} wrap {id}"))?;
                let key = secret_wrap_key(provider, secret)?;
                (index, unwrap_key(&key, &self.kek_wraps[index], KEK_AAD)?)
            }
        };
        if b64e(&fingerprint(kek.as_slice())) != self.kek_fingerprint {
            return Err(anyhow!(
                "{} unsealed an unexpected key (keystore fingerprint mismatch)",
                factor.provider()
            ));
        }
        Ok((index, kek))
    }

    fn check_kek(&self, kek: &SecretKey) -> Result<()> {
        if b64e(&fingerprint(kek.as_slice())) != self.kek_fingerprint {
            return Err(anyhow!("that is not this keystore's master key"));
        }
        Ok(())
    }

    /// Whether `factor` has a wrap here to open, without trying to open it.
    fn has_wrap_for(&self, factor: &Factor) -> bool {
        match factor {
            Factor::RecoveryCode(_) => self
                .kek_wraps
                .iter()
                .any(|w| w.provider == PROVIDER_RECOVERY),
            Factor::Secret { provider, id, .. } => self
                .kek_wraps
                .iter()
                .any(|w| w.provider == *provider && wrap_id(w) == *id),
            other => self
                .kek_wraps
                .iter()
                .any(|w| w.provider == other.provider()),
        }
    }

    /// `kek` wrapped under `factor`, with the KDF parameters of a passphrase wrap.
    fn wrap_under(factor: &Factor, kek: &SecretKey) -> Result<(Wrapped, Option<KdfParams>)> {
        Ok(match factor {
            Factor::Passphrase(passphrase) => {
                let kdf = KdfParams::production();
                let unseal = kdf.derive(passphrase)?;
                let wrap = wrap_key(&unseal, kek.as_slice(), KEK_AAD, PROVIDER_PASSPHRASE)?;
                (wrap, Some(kdf))
            }
            Factor::RecoveryCode(code) => {
                let mut wrap = wrap_key(
                    &recovery_key_from_code(code),
                    kek.as_slice(),
                    KEK_AAD,
                    PROVIDER_RECOVERY,
                )?;
                wrap.id = Some(recovery_code_id(code));
                (wrap, None)
            }
            Factor::Kms(kms) => (
                Wrapped {
                    provider: PROVIDER_KMS.to_string(),
                    id: None,
                    label: None,
                    created_at: None,
                    nonce: String::new(),
                    ciphertext: b64e(&kms.kms_encrypt(kek.as_slice())?),
                },
                None,
            ),
            Factor::Secret {
                provider,
                id,
                secret,
            } => {
                let key = secret_wrap_key(provider, secret)?;
                let mut wrap = wrap_key(&key, kek.as_slice(), KEK_AAD, provider)?;
                wrap.id = Some(id.to_string());
                (wrap, None)
            }
        })
    }

    /// Enrol `factor` as a further way to unwrap the master key. A passphrase, KMS or
    /// ring wrap replaces the one before it, a device factor replaces only its own id,
    /// and recovery codes accumulate until revoked.
    pub fn add_factor(
        &mut self,
        kek: &SecretKey,
        factor: &Factor,
        label: Option<&str>,
        created_at: &str,
    ) -> Result<FactorInfo> {
        self.check_kek(kek)?;
        let (mut wrap, kdf) = Self::wrap_under(factor, kek)?;
        wrap.label = label.map(str::to_string);
        wrap.created_at = Some(created_at.to_string());
        let provider = factor.provider().to_string();
        let id = wrap_id(&wrap).to_string();
        let single =
            [PROVIDER_PASSPHRASE, PROVIDER_KMS, PROVIDER_RING].contains(&provider.as_str());
        self.kek_wraps
            .retain(|w| w.provider != provider || (!single && wrap_id(w) != id));
        let info = factor_info(&wrap);
        self.kek_wraps.push(wrap);
        if let Some(kdf) = kdf {
            self.kdf = kdf;
        }
        self.generation += 1;
        self.seal(kek)?;
        Ok(info)
    }

    /// Remove the wrap `id`. The passphrase wrap is rotated, never revoked: it is how
    /// the server unseals at start. A keystore is never left without a way in.
    pub fn revoke_factor(&mut self, kek: &SecretKey, id: &str) -> Result<FactorInfo> {
        self.check_kek(kek)?;
        let index = self
            .kek_wraps
            .iter()
            .position(|w| wrap_id(w) == id)
            .ok_or_else(|| anyhow!("this keystore has no factor {id}"))?;
        if self.kek_wraps[index].provider == PROVIDER_PASSPHRASE {
            return Err(anyhow!("the passphrase cannot be revoked, only rotated"));
        }
        if self.kek_wraps.len() == 1 {
            return Err(anyhow!("{id} is the only way into this keystore"));
        }
        let removed = self.kek_wraps.remove(index);
        self.generation += 1;
        self.seal(kek)?;
        Ok(factor_info(&removed))
    }

    /// The key purpose keys and the index blinder derive from: the master key until
    /// the first master-key rotation, then the original master key, wrapped under the
    /// current one.
    pub fn root_key(&self, kek: &SecretKey) -> Result<SecretKey> {
        match &self.root_wrap {
            Some(wrap) => unwrap_key(kek, wrap, ROOT_AAD),
            None => Ok(kek.clone()),
        }
    }

    /// The `purpose` key derived from this keystore's root.
    pub fn purpose_cryptor(&self, kek: &SecretKey, purpose: &str) -> Result<SecretCryptor> {
        Ok(SecretCryptor::new(derive_secret_key(
            &self.root_key(kek)?,
            purpose,
        )))
    }

    /// The key this keystore, as a keyring, lends the keystores it opens, and the id
    /// their ring wraps carry. Derived from the root, so it survives rotation.
    pub fn ring_key(&self, kek: &SecretKey) -> Result<(String, SecretKey)> {
        let secret = derive_secret_key(&self.root_key(kek)?, "keyring:ring");
        let id = format!("ring-{}", hex::encode(fingerprint(secret.as_slice())));
        Ok((id, secret))
    }

    /// Replace the master key. Data keys and the root are re-wrapped under the new
    /// one, so records, purpose keys and blinded index terms are untouched. Each
    /// factor in `keep` that has a wrap here is re-wrapped under the new key with its
    /// id and label; every other wrap is dropped and returned. Nothing changes on
    /// error. Returns the new master key.
    pub fn rotate_kek(
        &mut self,
        kek: &SecretKey,
        keep: &[Factor],
    ) -> Result<(SecretKey, Vec<FactorInfo>)> {
        self.verify_integrity(kek)?;
        let root = self.root_key(kek)?;
        let new = random_key();
        let mut kept = Vec::new();
        let mut kept_index = Vec::new();
        let mut kdf = None;
        for factor in keep.iter().filter(|f| self.has_wrap_for(f)) {
            let (index, _) = self.open_wrap(factor)?;
            if kept_index.contains(&index) {
                continue;
            }
            let (mut wrap, params) = Self::wrap_under(factor, &new)?;
            let old = &self.kek_wraps[index];
            wrap.id = old.id.clone();
            wrap.label = old.label.clone();
            wrap.created_at = old.created_at.clone();
            kept.push(wrap);
            kept_index.push(index);
            kdf = params.or(kdf);
        }
        if kept.is_empty() {
            return Err(anyhow!(
                "none of the factors to keep has a wrap in this keystore; it would have no way in"
            ));
        }
        let dropped = self
            .kek_wraps
            .iter()
            .enumerate()
            .filter(|(i, _)| !kept_index.contains(i))
            .map(|(_, w)| factor_info(w))
            .collect();
        let deks = self
            .deks
            .iter()
            .map(|d| {
                let dek = self.data_key_for_epoch(kek, d.epoch)?;
                Ok(DekEntry {
                    epoch: d.epoch,
                    wrapped: wrap_key(&new, dek.as_slice(), &dek_aad(d.epoch), "kek")?,
                    state: d.state.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let root_wrap = wrap_key(&new, root.as_slice(), ROOT_AAD, "kek")?;

        self.kek_wraps = kept;
        self.deks = deks;
        self.root_wrap = Some(root_wrap);
        if let Some(kdf) = kdf {
            self.kdf = kdf;
        }
        self.kek_fingerprint = b64e(&fingerprint(new.as_slice()));
        self.generation += 1;
        self.seal(&new)?;
        Ok((new, dropped))
    }

    /// A keystore whose only way in is `factor`, for a tenant created while the
    /// server holds a keyring but no passphrase.
    pub fn create_with_factor(factor: &Factor, created_at: &str) -> Result<Self> {
        let kek = random_key();
        let dek = random_key();
        let (mut wrap, kdf) = Self::wrap_under(factor, &kek)?;
        wrap.created_at = Some(created_at.to_string());
        let mut ks = Self {
            crypto_version: CRYPTO_VERSION,
            schema_version: SCHEMA_VERSION,
            kdf: kdf.unwrap_or_else(KdfParams::production),
            kek_wraps: vec![wrap],
            deks: vec![DekEntry {
                epoch: 0,
                wrapped: wrap_key(&kek, dek.as_slice(), &dek_aad(0), "kek")?,
                state: "active".to_string(),
            }],
            active_epoch: 0,
            kek_fingerprint: b64e(&fingerprint(kek.as_slice())),
            root_wrap: None,
            generation: 0,
            mac: String::new(),
        };
        ks.seal(&kek)?;
        Ok(ks)
    }

    /// Atomically persist the keystore to `path`: write a temp file, fsync it,
    /// then rename over the target, so a crash/power-loss never leaves a truncated
    /// or empty keystore (which would make the KEK — and thus all data —
    /// unrecoverable). Keeps a `.bak` of the prior keystore as a recovery copy.
    /// On unix the file is chmod 0600 and the parent dir is fsynced so the rename
    /// is durable.
    pub fn save_to_path(&self, path: &Path) -> Result<()> {
        use std::io::Write;

        let json = self.to_json()?;

        // Preserve the previous keystore as a recovery backup before overwriting.
        if path.exists() {
            let bak = path.with_extension("json.bak");
            std::fs::copy(path, &bak).context("failed to back up existing keystore")?;
        }

        let tmp = path.with_extension("json.tmp");
        {
            let mut f =
                std::fs::File::create(&tmp).context("failed to create keystore temp file")?;
            f.write_all(json.as_bytes())
                .context("failed to write keystore temp file")?;
            f.sync_all().context("failed to fsync keystore temp file")?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
        }
        std::fs::rename(&tmp, path).context("failed to rename keystore into place")?;
        // fsync the directory so the rename itself is durable across a crash.
        #[cfg(unix)]
        if let Some(dir) = path.parent() {
            if let Ok(d) = std::fs::File::open(dir) {
                let _ = d.sync_all();
            }
        }
        Ok(())
    }

    /// Open an existing keystore at `path` (verifying its integrity MAC) or, if
    /// absent, create + atomically persist one. Unseals with `passphrase` and
    /// returns the keystore + master key. Used by standalone secret encryptors
    /// (slow-store / TOTP) that share one keystore file but no database, so there
    /// is no DB rollback sentinel here — file-integrity (MAC) only.
    /// Persist like [`save_to_path`](Self::save_to_path), then replace the `.bak` with
    /// the same content: after a revocation or a master-key rotation the previous file
    /// still opens under what was removed.
    pub fn save_superseding(&self, path: &Path) -> Result<()> {
        self.save_to_path(path)?;
        let bak = path.with_extension("json.bak");
        let tmp = path.with_extension("json.bak.tmp");
        std::fs::copy(path, &tmp).context("failed to copy the keystore over its backup")?;
        std::fs::File::open(&tmp)
            .and_then(|f| f.sync_all())
            .context("failed to fsync the keystore backup")?;
        std::fs::rename(&tmp, &bak).context("failed to replace the keystore backup")?;
        Ok(())
    }

    pub fn open_or_create_at(path: &Path, passphrase: &str) -> Result<(Self, SecretKey)> {
        if path.exists() {
            let ks = Self::from_json(
                &std::fs::read_to_string(path).context("failed to read keystore")?,
            )?;
            let kek = ks.unseal_with_passphrase(passphrase)?;
            ks.verify_integrity(&kek)?;
            Ok((ks, kek))
        } else {
            let ks = Self::create(passphrase, KdfParams::production())?;
            ks.save_to_path(path)?;
            let kek = ks.unseal_with_passphrase(passphrase)?;
            Ok((ks, kek))
        }
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("failed to serialize keystore")
    }

    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("failed to parse keystore")
    }
}

type HmacSha256 = Hmac<Sha256>;

/// Label for deriving the (stable) index-blinding key from the master key.
const INDEX_KEY_LABEL: &[u8] = b"veld:index-blind:v1";

/// Derives deterministic index-blinding tokens: `hex(HMAC-SHA256(index_key, term))`
/// (P3). Equal terms map to equal tokens (so exact-match index lookups still work)
/// while the plaintext term stays off disk. Leaks term equality/frequency only
/// (design §6 / §11 B-4). The key is derived from the keystore's root, which a
/// master-key rotation keeps, so the index survives one.
pub struct IndexBlinder {
    key: SecretKey,
}

impl IndexBlinder {
    pub fn new(key: SecretKey) -> Self {
        Self { key }
    }

    /// Derive the stable index-blinding key from the keystore's root
    /// ([`Keystore::root_key`]).
    pub fn derive_from_root(root: &SecretKey) -> Self {
        let mut mac = <HmacSha256 as Mac>::new_from_slice(root.as_slice())
            .expect("HMAC accepts any key length");
        Mac::update(&mut mac, INDEX_KEY_LABEL);
        let tag = Mac::finalize(mac).into_bytes();
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        key.copy_from_slice(&tag[..KEY_LEN]);
        Self::new(key)
    }

    /// Blind a single index term to a hex token (safe for ':'-delimited index keys).
    pub fn blind(&self, term: &str) -> String {
        let mut mac = <HmacSha256 as Mac>::new_from_slice(self.key.as_slice())
            .expect("HMAC accepts any key length");
        Mac::update(&mut mac, term.as_bytes());
        hex::encode(Mac::finalize(mac).into_bytes())
    }
}

// ============================================================================
// Record envelope (P2) — encrypt/decrypt a serialized record blob under a DEK.
// Wire format: `ENC\0 | crypto_version(1) | epoch(4 LE) | nonce(24) | ct+tag`.
// Records use XChaCha20-Poly1305 (192-bit nonce) so random nonces never approach
// a collision bound; key-wrapping (low volume) stays on AES-256-GCM.
// AAD binds the epoch (prevents cross-epoch substitution). Record-id binding is
// a documented follow-up: it requires confirming the RocksDB key == memory id
// invariant at every write site first (design §5).
// ============================================================================

/// 4-byte marker distinguishing an encrypted record from a legacy plaintext
/// (bincode) record. bincode of a `Memory` never begins with `ENC\0`.
const RECORD_MARKER: &[u8; 4] = b"ENC\x00";
/// marker(4) + version(1) + epoch(4) + nonce(24, XChaCha20)
const RECORD_HEADER_LEN: usize = 4 + 1 + 4 + 24;
/// Poly1305 tag is 16 bytes; a valid ciphertext is at least the tag.
const RECORD_MIN_LEN: usize = RECORD_HEADER_LEN + 16;

/// Associated data for record AEAD: domain string + epoch, plus a caller-supplied
/// identity context (e.g. the memory-id for records, the user-id for TOTP secrets)
/// so a ciphertext only authenticates under the identity it was written for —
/// preventing cross-record / cross-user blob swaps by an attacker with datastore
/// write access. An empty `ctx` means "epoch-bound only" (no identity binding).
fn record_aad(epoch: u32, ctx: &[u8]) -> Vec<u8> {
    let mut aad = format!("veld:record:v1:epoch:{epoch}").into_bytes();
    aad.push(0x1f); // unit separator: keep ctx from colliding with the epoch text
    aad.extend_from_slice(ctx);
    aad
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

/// A keystore unsealed for record-level encryption: one cryptor per epoch, the
/// index blinder, and the generation a store's rollback sentinel is checked against.
pub struct UnsealedRecordKeys {
    pub cryptors: RecordCryptors,
    pub blinder: IndexBlinder,
    pub generation: u64,
}

/// Where a tenant's record store keeps its keystore: beside `storage.redb`, or
/// inside the RocksDB `storage/` directory.
pub fn store_keystore_path(
    user_dir: &std::path::Path,
    backend: crate::config::StorageBackend,
) -> std::path::PathBuf {
    match backend {
        crate::config::StorageBackend::RocksDb => user_dir.join("storage").join("keystore.json"),
        crate::config::StorageBackend::Redb => user_dir.join("keystore.json"),
    }
}

// Serialises open-or-create of a store keystore. The record store and the journal of
// the same tenant both unseal it; on a first run with a passphrase each would otherwise
// create one, and records sealed under the overwritten keystore's key would be lost.
static RECORD_KEYSTORE_OPEN: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Unseal — or, on a first run with a passphrase, create — the store keystore at
/// `keystore_path`. `Ok(None)` only when there is no keystore and no passphrase.
fn unseal_record_keystore(
    keystore_path: &std::path::Path,
) -> Result<Option<(Keystore, SecretKey)>> {
    let _guard = RECORD_KEYSTORE_OPEN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let passphrase = std::env::var("VELD_MASTER_PASSPHRASE")
        .ok()
        .filter(|s| !s.is_empty());
    let ring = crate::keyring::installed();
    let exists = keystore_path
        .try_exists()
        .with_context(|| format!("Failed to stat {keystore_path:?}"))?;

    let (ks, kek) = if exists {
        let json =
            std::fs::read_to_string(keystore_path).context("Failed to read keystore.json")?;
        let ks = Keystore::from_json(&json)?;
        let kek = match (&ring, &passphrase) {
            (Some(ring), _) if ks.has_ring(ring.id()) => {
                let kek = ks.unseal(&ring.factor())?;
                tracing::info!("Record-level encryption: keystore unsealed (keyring)");
                kek
            }
            (_, Some(pass)) => {
                let kek = ks.unseal_with_passphrase(pass)?;
                tracing::info!("Record-level encryption: keystore unsealed (passphrase)");
                kek
            }
            (_, None) => {
                let kms = LocalAeadKms::from_env()?.ok_or_else(|| {
                    anyhow!(
                        "{keystore_path:?} is present but neither VELD_MASTER_PASSPHRASE nor \
                         VELD_KMS_WRAP_KEY is set and the keyring does not open it; refusing to \
                         start (encrypted records would be served as ciphertext)"
                    )
                })?;
                let kek = ks.unseal_with_kms(&kms)?;
                tracing::info!("Record-level encryption: keystore unsealed (KMS)");
                kek
            }
        };
        (ks, kek)
    } else {
        let ks = match (&passphrase, &ring) {
            (Some(pass), _) => Keystore::create(pass, KdfParams::production())?,
            (None, Some(ring)) => {
                Keystore::create_with_factor(&ring.factor(), &crate::keyring::now_rfc3339())?
            }
            (None, None) => return Ok(None),
        };
        // The journal can open first on a fresh tenant, before the store made
        // its directory.
        if let Some(dir) = keystore_path.parent() {
            std::fs::create_dir_all(dir).context("Failed to create the keystore directory")?;
        }
        ks.save_to_path(keystore_path)
            .context("Failed to write keystore.json")?;
        let kek = match (&passphrase, &ring) {
            (Some(pass), _) => ks.unseal_with_passphrase(pass)?,
            (None, Some(ring)) => ks.unseal(&ring.factor())?,
            (None, None) => unreachable!("returned above"),
        };
        tracing::info!("Created new encryption keystore at {keystore_path:?}");
        (ks, kek)
    };

    ks.verify_integrity(&kek)
        .context("keystore integrity verification failed")?;
    let ks = match &ring {
        Some(ring) if !ks.has_ring(ring.id()) => enrol_in_ring(keystore_path, ks, &kek, ring)?,
        _ => ks,
    };
    Ok(Some((ks, kek)))
}

/// Give a keystore opened another way its ring wrap, so the keyring's factors reach
/// it. Re-read under the lock: another writer may have replaced the file meanwhile,
/// and one whose master key has since changed is left for the next start.
fn enrol_in_ring(
    path: &std::path::Path,
    opened: Keystore,
    kek: &SecretKey,
    ring: &crate::keyring::RingKey,
) -> Result<Keystore> {
    let _lock = crate::keyring::lock_keystore(path)?;
    let mut current = Keystore::from_json(
        &std::fs::read_to_string(path).context("Failed to read keystore.json")?,
    )?;
    if current.verify_integrity(kek).is_err() {
        tracing::warn!("{path:?} changed while it was opened; it joins the keyring next start");
        return Ok(opened);
    }
    if current.has_ring(ring.id()) {
        return Ok(current);
    }
    current.add_factor(kek, &ring.factor(), None, &crate::keyring::now_rfc3339())?;
    current
        .save_to_path(path)
        .context("Failed to write keystore.json")?;
    tracing::info!("{path:?} enrolled in the keyring");
    Ok(current)
}

/// Open the keystore at `keystore_path` for a store's records, on either engine.
///
/// Unsealed with `VELD_MASTER_PASSPHRASE`, or `VELD_KMS_WRAP_KEY` when no
/// passphrase is set. With a passphrase and no keystore, one is created. `None`
/// when there is neither a keystore nor a passphrase: a plaintext store. A
/// keystore present but not unsealable is an error, so ciphertext is never served
/// as plaintext. Integrity is verified before the keys are returned.
pub fn open_record_keys(keystore_path: &std::path::Path) -> Result<Option<UnsealedRecordKeys>> {
    let Some((ks, kek)) = unseal_record_keystore(keystore_path)? else {
        return Ok(None);
    };
    record_keys(&ks, &kek).map(Some)
}

/// Unseal an existing keystore's JSON for record-level encryption: through the
/// keyring when the keystore carries its ring wrap, otherwise with
/// `VELD_MASTER_PASSPHRASE`, or `VELD_KMS_WRAP_KEY` when no passphrase is set.
/// Never creates a keystore.
pub fn unseal_record_keys(json: &str) -> Result<UnsealedRecordKeys> {
    let ks = Keystore::from_json(json)?;
    let ring = crate::keyring::installed();
    let passphrase = std::env::var("VELD_MASTER_PASSPHRASE")
        .ok()
        .filter(|s| !s.is_empty());
    let kek = match (&ring, passphrase) {
        (Some(ring), _) if ks.has_ring(ring.id()) => {
            let kek = ks.unseal(&ring.factor())?;
            tracing::info!("Record-level encryption: keystore unsealed (keyring)");
            kek
        }
        (_, Some(pass)) => {
            let kek = ks.unseal_with_passphrase(&pass)?;
            tracing::info!("Record-level encryption: keystore unsealed (passphrase)");
            kek
        }
        (_, None) => {
            let kms = LocalAeadKms::from_env()?.ok_or_else(|| {
                anyhow!(
                    "keystore.json is present but neither VELD_MASTER_PASSPHRASE nor \
                     VELD_KMS_WRAP_KEY is set and the keyring does not open it; refusing to \
                     start (encrypted records would be served as ciphertext)"
                )
            })?;
            let kek = ks.unseal_with_kms(&kms)?;
            tracing::info!("Record-level encryption: keystore unsealed (KMS)");
            kek
        }
    };
    record_keys(&ks, &kek)
}

fn record_keys(ks: &Keystore, kek: &SecretKey) -> Result<UnsealedRecordKeys> {
    ks.verify_integrity(kek)
        .context("keystore integrity verification failed")?;
    Ok(UnsealedRecordKeys {
        cryptors: RecordCryptors::from_keystore(ks, kek)?,
        blinder: IndexBlinder::derive_from_root(&ks.root_key(kek)?),
        generation: ks.generation,
    })
}

/// Where both engines keep the last keystore generation they saw: RocksDB in its
/// `memory_index` family, redb in its `default` table.
pub const GENERATION_SENTINEL_KEY: &[u8] = b"meta:keystore_generation";

/// The rollback guard's decision for a store whose sentinel holds `stored` and
/// whose keystore file is at `generation`. `Ok(Some(g))`: write `g` to the
/// sentinel. `Ok(None)`: leave it. An older keystore is refused unless
/// `VELD_ALLOW_KEYSTORE_ROLLBACK` accepts it for this start.
pub fn keystore_generation_sentinel(stored: u64, generation: u64) -> Result<Option<u64>> {
    if generation < stored {
        let allow_rollback = std::env::var("VELD_ALLOW_KEYSTORE_ROLLBACK")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "true" || v == "1"
            })
            .unwrap_or(false);
        if !allow_rollback {
            return Err(anyhow!(
                "keystore rollback detected: file generation {generation} < last-seen {stored}. \
                 If you intentionally restored an older keystore (e.g. keystore.json.bak), set \
                 VELD_ALLOW_KEYSTORE_ROLLBACK=true once to accept it and reset the sentinel."
            ));
        }
        tracing::warn!(
            file_generation = generation,
            last_seen = stored,
            "VELD_ALLOW_KEYSTORE_ROLLBACK set: accepting an older keystore generation (restored \
             keystore?) and resetting the rollback sentinel to it. Rollback protection is \
             bypassed for this start — unset the variable afterward."
        );
        return Ok(Some(generation));
    }
    Ok((generation > stored).then_some(generation))
}

/// Decrypt an encrypted record blob with the DEK for its epoch. The caller is
/// responsible for selecting `dek` matching [`record_epoch`].
pub fn decrypt_record(dek: &SecretKey, data: &[u8], aad_ctx: &[u8]) -> Result<Vec<u8>> {
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
    let nonce = XNonce::from_slice(&data[9..RECORD_HEADER_LEN]);
    let ciphertext = &data[RECORD_HEADER_LEN..];
    let cipher = XChaCha20Poly1305::new(chacha20poly1305::Key::from_slice(dek.as_slice()));
    cipher
        .decrypt(
            nonce,
            Payload {
                msg: ciphertext,
                aad: &record_aad(epoch, aad_ctx),
            },
        )
        .map_err(|e| anyhow!("record decrypt failed (wrong key/epoch/identity or tampered): {e}"))
}

/// Encrypts serialized record blobs under a fixed active DEK/epoch. Held by the
/// storage layer (P2 wiring). Reads of older epochs go through [`decrypt_record`]
/// with the DEK fetched from the keystore for that epoch.
#[derive(Clone)]
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

    /// Encrypt a serialized record blob under the active DEK. `aad_ctx` binds the
    /// ciphertext to a caller identity (memory-id / user-id); pass `&[]` for no
    /// identity binding. The same `aad_ctx` must be supplied to decrypt.
    pub fn encrypt_record(&self, plaintext: &[u8], aad_ctx: &[u8]) -> Result<Vec<u8>> {
        let cipher = XChaCha20Poly1305::new(chacha20poly1305::Key::from_slice(self.dek.as_slice()));
        let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(
                &nonce,
                Payload {
                    msg: plaintext,
                    aad: &record_aad(self.epoch, aad_ctx),
                },
            )
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
    /// `aad_ctx` must match what was passed to [`encrypt_record`](Self::encrypt_record).
    pub fn decrypt_record(&self, data: &[u8], aad_ctx: &[u8]) -> Result<Vec<u8>> {
        decrypt_record(&self.dek, data, aad_ctx)
    }
}

/// The set of record cryptors a store needs: the active one for new writes, plus
/// one per retired epoch so records written before a key rotation stay readable.
/// Built from the keystore's retained per-epoch DEKs — this is what makes data
/// key rotation safe (the read path selects the DEK for each record's epoch).
#[derive(Clone)]
pub struct RecordCryptors {
    active_epoch: u32,
    by_epoch: std::collections::HashMap<u32, RecordCryptor>,
}

impl RecordCryptors {
    /// Derive a cryptor for every epoch the keystore retains a DEK for.
    pub fn from_keystore(ks: &Keystore, kek: &SecretKey) -> Result<Self> {
        let mut by_epoch = std::collections::HashMap::new();
        for entry in &ks.deks {
            let dek = ks.data_key_for_epoch(kek, entry.epoch)?;
            by_epoch.insert(entry.epoch, RecordCryptor::new(entry.epoch, dek));
        }
        if !by_epoch.contains_key(&ks.active_epoch) {
            return Err(anyhow!(
                "keystore active_epoch {} has no DEK entry",
                ks.active_epoch
            ));
        }
        Ok(Self {
            active_epoch: ks.active_epoch,
            by_epoch,
        })
    }

    /// Cryptor for new writes (the active epoch).
    pub fn active(&self) -> &RecordCryptor {
        self.by_epoch
            .get(&self.active_epoch)
            .expect("active-epoch cryptor present (checked at construction)")
    }

    /// Cryptor for a stored record's epoch, if the keystore retains that DEK.
    pub fn for_epoch(&self, epoch: u32) -> Option<&RecordCryptor> {
        self.by_epoch.get(&epoch)
    }

    pub fn active_epoch(&self) -> u32 {
        self.active_epoch
    }
}

/// Derive a purpose-bound 256-bit secret key from the keystore's root:
/// `HMAC-SHA256(root, "veld:secret:v1:" || purpose)[..32]`. The root is the KEK
/// until the first master-key rotation. Domain separation
/// keeps each subsystem's key (slow-store, totp, …) distinct from each other and
/// from the record DEK / index-blinding key.
fn derive_secret_key(kek: &SecretKey, purpose: &str) -> SecretKey {
    let mut mac =
        <HmacSha256 as Mac>::new_from_slice(kek.as_slice()).expect("HMAC accepts any key length");
    Mac::update(&mut mac, b"veld:secret:v1:");
    Mac::update(&mut mac, purpose.as_bytes());
    let tag = Mac::finalize(mac).into_bytes();
    let mut key = Zeroizing::new([0u8; KEY_LEN]);
    key.copy_from_slice(&tag[..KEY_LEN]);
    key
}

/// Purpose-bound secret-blob encryptor (encryption-v2) for non-record secrets
/// such as the slow-store memory payload and TOTP secrets. A drop-in for the
/// legacy v1 field-encryptor API (`encrypt_bytes` / `decrypt_bytes` / `is_encrypted`),
/// but XChaCha20-Poly1305 keyed by a keystore-derived, purpose-bound key — so it
/// inherits the keystore's atomic persistence, integrity MAC, and rotation
/// instead of a bare single `VELD_ENCRYPTION_KEY`.
#[derive(Clone)]
pub struct SecretCryptor {
    inner: RecordCryptor,
}

impl SecretCryptor {
    /// Construct from a raw purpose key (tests / explicit injection).
    pub fn new(key: SecretKey) -> Self {
        Self {
            inner: RecordCryptor::new(0, key),
        }
    }

    /// Build from the keystore at `VELD_KEYSTORE_PATH`, unsealed with
    /// `VELD_MASTER_PASSPHRASE`, deriving a `purpose`-bound key.
    ///
    /// Returns `Ok(None)` only when `VELD_KEYSTORE_PATH` is unset — no encryption
    /// configured, plaintext, backward compatible. If the keystore path *is* set
    /// but `VELD_MASTER_PASSPHRASE` is missing/empty this is a hard error, not
    /// silent plaintext: it mirrors the primary record store, which refuses to
    /// start a configured-but-unsealable keystore rather than write data that was
    /// meant to be encrypted as plaintext-at-rest. The keystore is created on
    /// first use.
    pub fn from_env(purpose: &str) -> Result<Option<Self>> {
        let path = match std::env::var("VELD_KEYSTORE_PATH") {
            Ok(p) if !p.trim().is_empty() => std::path::PathBuf::from(p),
            _ => return Ok(None),
        };
        // The keystore path is set, so encryption is intended. A missing
        // passphrase must fail closed — never degrade to plaintext-at-rest.
        let pass = match std::env::var("VELD_MASTER_PASSPHRASE") {
            Ok(p) if !p.is_empty() => p,
            _ => anyhow::bail!(
                "VELD_KEYSTORE_PATH is set but VELD_MASTER_PASSPHRASE is unset/empty; \
                 refusing to store '{purpose}' secrets as plaintext"
            ),
        };
        let (ks, kek) = Keystore::open_or_create_at(&path, &pass)?;
        Ok(Some(Self::new(derive_secret_key(
            &ks.root_key(&kek)?,
            purpose,
        ))))
    }

    /// A `purpose`-bound key derived from the tenant's STORE keystore, the one its
    /// sealed records use. Without `VELD_KEYSTORE_PATH`, records are sealed under
    /// this keystore alone, and a journal or secret left unsealed beside them would
    /// be a plaintext copy of what the records protect: one keystore, one decision.
    /// `Ok(None)` when the store has no keystore and no passphrase is set.
    pub fn from_record_keystore(
        keystore_path: &std::path::Path,
        purpose: &str,
    ) -> Result<Option<Self>> {
        let Some((ks, kek)) = unseal_record_keystore(keystore_path)? else {
            return Ok(None);
        };
        Ok(Some(Self::new(derive_secret_key(
            &ks.root_key(&kek)?,
            purpose,
        ))))
    }

    /// Encrypt a secret blob. `aad_ctx` binds it to a caller identity (e.g. the
    /// user-id for a TOTP secret, the memory-id for a slow-store payload); pass
    /// `&[]` for no binding. The same `aad_ctx` must be supplied to decrypt.
    pub fn encrypt_bytes(&self, plaintext: &[u8], aad_ctx: &[u8]) -> Result<Vec<u8>> {
        self.inner.encrypt_record(plaintext, aad_ctx)
    }

    pub fn decrypt_bytes(&self, data: &[u8], aad_ctx: &[u8]) -> Result<Vec<u8>> {
        self.inner.decrypt_record(data, aad_ctx)
    }

    /// UTF-8 convenience wrapper over [`encrypt_bytes`](Self::encrypt_bytes).
    pub fn encrypt_content(&self, plaintext: &str, aad_ctx: &[u8]) -> Result<Vec<u8>> {
        self.encrypt_bytes(plaintext.as_bytes(), aad_ctx)
    }

    /// UTF-8 convenience wrapper over [`decrypt_bytes`](Self::decrypt_bytes).
    pub fn decrypt_content(&self, data: &[u8], aad_ctx: &[u8]) -> Result<String> {
        String::from_utf8(self.decrypt_bytes(data, aad_ctx)?)
            .context("decrypted secret is not valid UTF-8")
    }

    /// `true` if `data` carries the encrypted-record marker.
    pub fn is_encrypted(data: &[u8]) -> bool {
        is_encrypted_record(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_unseal_roundtrip() {
        let ks =
            Keystore::create("correct horse battery staple", KdfParams::fast_for_tests()).unwrap();
        let kek = ks
            .unseal_with_passphrase("correct horse battery staple")
            .unwrap();
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
    fn derive_rejects_excessive_argon2_params() {
        // A tampered keystore with absurd m_cost must be rejected before Argon2 runs.
        let kdf = KdfParams {
            m_cost: u32::MAX,
            t_cost: 3,
            p_cost: 1,
            salt: b64e(&[0u8; SALT_LEN]),
        };
        assert!(kdf.derive("pw").is_err());
    }

    #[test]
    fn derive_rejects_weak_argon2_params() {
        // A tampered keystore that downgrades the KDF below the floor must be
        // rejected before Argon2 runs (prevents an offline brute-force downgrade).
        let kdf = KdfParams {
            m_cost: 8, // the old absolute-minimum; now below MIN_ARGON2_M_COST_KIB
            t_cost: 1,
            p_cost: 1,
            salt: b64e(&[0u8; SALT_LEN]),
        };
        assert!(kdf.derive("pw").is_err());
    }

    #[test]
    fn record_cryptors_decrypt_across_epochs_after_rotation() {
        let mut ks = Keystore::create("pw-rotation-test", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw-rotation-test").unwrap();

        // Encrypt a record under the original epoch (0).
        let pre = RecordCryptors::from_keystore(&ks, &kek).unwrap();
        assert_eq!(pre.active_epoch(), 0);
        let ct0 = pre
            .active()
            .encrypt_record(b"pre-rotation-blob", b"")
            .unwrap();
        assert_eq!(record_epoch(&ct0), Some(0));

        // Rotate the DEK -> epoch 1.
        assert_eq!(ks.rotate_dek(&kek).unwrap(), 1);

        // A fresh cryptor set (as a restarted process would build) reads BOTH epochs.
        let post = RecordCryptors::from_keystore(&ks, &kek).unwrap();
        assert_eq!(post.active_epoch(), 1, "writes now use epoch 1");
        // The pre-rotation record still decrypts under its retired epoch-0 DEK.
        assert_eq!(
            post.for_epoch(0)
                .unwrap()
                .decrypt_record(&ct0, b"")
                .unwrap(),
            b"pre-rotation-blob"
        );
        // New writes go out at epoch 1 and round-trip.
        let ct1 = post
            .active()
            .encrypt_record(b"post-rotation-blob", b"")
            .unwrap();
        assert_eq!(record_epoch(&ct1), Some(1));
        assert_eq!(
            post.for_epoch(1)
                .unwrap()
                .decrypt_record(&ct1, b"")
                .unwrap(),
            b"post-rotation-blob"
        );
    }

    #[test]
    fn recovery_code_resets_lost_passphrase() {
        let mut ks = Keystore::create("original-pass", KdfParams::fast_for_tests()).unwrap();
        let kek0 = ks.unseal_with_passphrase("original-pass").unwrap();
        let code = ks.add_recovery_code(&kek0).unwrap();

        // Passphrase "lost": recover via the code, then install a new passphrase.
        let kek_via_code = ks.unseal_with_recovery_code(&code).unwrap();
        ks.set_passphrase(&kek_via_code, "brand-new-pass").unwrap();

        // The new passphrase unseals the SAME master key...
        let kek1 = ks.unseal_with_passphrase("brand-new-pass").unwrap();
        assert_eq!(kek0.as_slice(), kek1.as_slice());
        // ...and the old passphrase no longer works.
        assert!(ks.unseal_with_passphrase("original-pass").is_err());
    }

    #[test]
    fn secret_cryptor_roundtrip_and_purpose_separation() {
        let kek = Zeroizing::new([6u8; KEY_LEN]);
        let slow = SecretCryptor::new(derive_secret_key(&kek, "slow-store"));
        let totp = SecretCryptor::new(derive_secret_key(&kek, "totp"));

        let ct = slow.encrypt_bytes(b"a-sensitive-blob", b"id-42").unwrap();
        assert!(SecretCryptor::is_encrypted(&ct));
        assert_ne!(ct, b"a-sensitive-blob");
        assert_eq!(
            slow.decrypt_bytes(&ct, b"id-42").unwrap(),
            b"a-sensitive-blob"
        );

        // Identity binding: decrypting under a different AAD context fails.
        assert!(slow.decrypt_bytes(&ct, b"id-99").is_err());
        // Domain separation: a different purpose derives a different key.
        assert!(totp.decrypt_bytes(&ct, b"id-42").is_err());
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
        let enc = cryptor.encrypt_record(&plaintext, b"mem-1").unwrap();
        assert!(is_encrypted_record(&enc));
        assert_eq!(record_epoch(&enc), Some(0));
        assert_ne!(enc, plaintext);
        assert_eq!(cryptor.decrypt_record(&enc, b"mem-1").unwrap(), plaintext);
        // Identity binding: the same ciphertext under a different id fails (anti-swap).
        assert!(cryptor.decrypt_record(&enc, b"mem-2").is_err());
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
        let enc = c1.encrypt_record(b"secret", b"").unwrap();
        assert!(decrypt_record(&Zeroizing::new([2u8; KEY_LEN]), &enc, b"").is_err());
    }

    #[test]
    fn record_tamper_is_detected() {
        let c = RecordCryptor::new(3, Zeroizing::new([7u8; KEY_LEN]));
        let mut enc = c.encrypt_record(b"payload", b"").unwrap();
        let last = enc.len() - 1;
        enc[last] ^= 0xff; // flip a ciphertext byte
        assert!(c.decrypt_record(&enc, b"").is_err());
    }

    #[test]
    fn record_epoch_in_header_matches() {
        let c = RecordCryptor::new(42, Zeroizing::new([9u8; KEY_LEN]));
        let enc = c.encrypt_record(b"x", b"").unwrap();
        assert_eq!(record_epoch(&enc), Some(42));
        assert_eq!(c.epoch(), 42);
    }

    #[test]
    fn index_blinder_is_deterministic_keyed_and_hides_term() {
        let kek = Zeroizing::new([4u8; KEY_LEN]);
        let b1 = IndexBlinder::derive_from_root(&kek);
        let b2 = IndexBlinder::derive_from_root(&kek);
        assert_eq!(b1.blind("alice"), b2.blind("alice")); // deterministic & stable
        assert_ne!(b1.blind("alice"), b1.blind("bob")); // distinct terms differ
        assert!(!b1.blind("alice").contains("alice")); // plaintext term not present
        assert_eq!(b1.blind("alice").len(), 64); // 32-byte HMAC as hex

        let other = IndexBlinder::derive_from_root(&Zeroizing::new([5u8; KEY_LEN]));
        assert_ne!(b1.blind("alice"), other.blind("alice")); // key-dependent
    }

    #[test]
    fn record_from_keystore_dek_end_to_end() {
        let ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        let (epoch, dek) = ks.active_data_key(&kek).unwrap();
        let cryptor = RecordCryptor::new(epoch, dek);
        let enc = cryptor.encrypt_record(b"end-to-end", b"").unwrap();
        assert_eq!(cryptor.decrypt_record(&enc, b"").unwrap(), b"end-to-end");
    }

    #[test]
    fn rotate_passphrase_preserves_master_key() {
        let mut ks = Keystore::create("old-pass", KdfParams::fast_for_tests()).unwrap();
        let kek_before = ks.unseal_with_passphrase("old-pass").unwrap();
        ks.rotate_passphrase("old-pass", "new-pass").unwrap();
        assert!(ks.unseal_with_passphrase("old-pass").is_err());
        let kek_after = ks.unseal_with_passphrase("new-pass").unwrap();
        // Same master key -> records/index untouched by a passphrase change.
        assert_eq!(kek_before.as_slice(), kek_after.as_slice());
    }

    #[test]
    fn rotate_passphrase_wrong_old_is_rejected() {
        let mut ks = Keystore::create("old", KdfParams::fast_for_tests()).unwrap();
        assert!(ks.rotate_passphrase("not-old", "new").is_err());
        assert!(ks.unseal_with_passphrase("old").is_ok()); // unchanged
    }

    #[test]
    fn recovery_code_unseals_same_key() {
        let mut ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        let code = ks.add_recovery_code(&kek).unwrap();
        assert_eq!(code.len(), 48);
        assert_eq!(
            ks.unseal_with_recovery_code(&code).unwrap().as_slice(),
            kek.as_slice()
        );
        assert!(ks.unseal_with_recovery_code("deadbeef").is_err());
    }

    #[test]
    fn recovery_survives_passphrase_rotation() {
        let mut ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        let code = ks.add_recovery_code(&kek).unwrap();
        ks.rotate_passphrase("pw", "pw2").unwrap();
        assert_eq!(
            ks.unseal_with_recovery_code(&code).unwrap().as_slice(),
            kek.as_slice()
        );
    }

    #[test]
    fn rotate_dek_adds_epoch_and_keeps_old_readable() {
        let mut ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        let (e0, dek0) = ks.active_data_key(&kek).unwrap();
        assert_eq!(e0, 0);

        let e1 = ks.rotate_dek(&kek).unwrap();
        assert_eq!(e1, 1);
        let (active_epoch, dek_active) = ks.active_data_key(&kek).unwrap();
        assert_eq!(active_epoch, 1);

        // Old epoch's DEK is still recoverable (for reading old records); new differs.
        assert_eq!(
            ks.data_key_for_epoch(&kek, 0).unwrap().as_slice(),
            dek0.as_slice()
        );
        assert_ne!(dek_active.as_slice(), dek0.as_slice());
    }

    #[test]
    fn local_kms_encrypt_decrypt_roundtrip() {
        let kms = LocalAeadKms::new([3u8; KEY_LEN]);
        let blob = kms.kms_encrypt(b"master-key-bytes").unwrap();
        assert_ne!(blob, b"master-key-bytes");
        assert_eq!(kms.kms_decrypt(&blob).unwrap(), b"master-key-bytes");
        assert!(LocalAeadKms::new([4u8; KEY_LEN])
            .kms_decrypt(&blob)
            .is_err());
    }

    #[test]
    fn kms_wrap_unseals_same_key_and_coexists_with_passphrase() {
        let mut ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        let kms = LocalAeadKms::new([9u8; KEY_LEN]);
        ks.add_kms_wrap(&kek, &kms).unwrap();
        assert_eq!(ks.unseal_with_kms(&kms).unwrap().as_slice(), kek.as_slice());
        assert!(ks.unseal_with_passphrase("pw").is_ok()); // multi-wrap: passphrase still works
        assert!(ks
            .unseal_with_kms(&LocalAeadKms::new([1u8; KEY_LEN]))
            .is_err()); // wrong KMS key
    }

    #[test]
    fn decode_32_accepts_hex_and_base64() {
        assert!(decode_32(&"00".repeat(32)).is_ok());
        assert!(decode_32(&base64::engine::general_purpose::STANDARD.encode([0u8; 32])).is_ok());
        assert!(decode_32("tooshort").is_err());
    }

    #[test]
    fn save_to_path_is_atomic_reloadable_and_backed_up() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keystore.json");
        let ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();

        ks.save_to_path(&path).unwrap();
        assert!(path.exists());
        assert!(
            !path.with_extension("json.tmp").exists(),
            "no leftover temp"
        );
        let loaded = Keystore::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(loaded.unseal_with_passphrase("pw").is_ok());

        // A second save preserves the prior keystore as a .bak recovery copy.
        ks.save_to_path(&path).unwrap();
        assert!(path.with_extension("json.bak").exists(), "backup written");
    }

    #[test]
    fn integrity_seal_detects_in_place_tampering() {
        let ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        ks.verify_integrity(&kek).unwrap(); // sealed at create

        let mut tampered = ks.clone();
        tampered.active_epoch = 99;
        assert!(tampered.verify_integrity(&kek).is_err());

        let mut t2 = ks.clone();
        t2.kek_fingerprint = "AAAAAA==".to_string();
        assert!(t2.verify_integrity(&kek).is_err());

        // survives a JSON round-trip (mac + generation persisted)
        let reloaded = Keystore::from_json(&ks.to_json().unwrap()).unwrap();
        reloaded.verify_integrity(&kek).unwrap();
    }

    #[test]
    fn generation_bumps_on_mutation_and_reseals() {
        let mut ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        assert_eq!(ks.generation, 0);
        let kek = ks.unseal_with_passphrase("pw").unwrap();

        ks.rotate_dek(&kek).unwrap();
        assert_eq!(ks.generation, 1);
        ks.verify_integrity(&kek).unwrap(); // re-sealed after mutation

        ks.add_recovery_code(&kek).unwrap();
        assert_eq!(ks.generation, 2);
        ks.verify_integrity(&kek).unwrap();
    }

    const NOW: &str = "2026-10-08T00:00:00Z";

    fn fresh(pass: &str) -> (Keystore, SecretKey) {
        let ks = Keystore::create(pass, KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase(pass).unwrap();
        (ks, kek)
    }

    #[test]
    fn several_recovery_codes_each_open_and_revoking_one_leaves_the_rest() {
        let (mut ks, kek) = fresh("pw");
        let (a, b) = (generate_recovery_code(), generate_recovery_code());
        let info_a = ks
            .add_factor(&kek, &Factor::RecoveryCode(&a), Some("safe"), NOW)
            .unwrap();
        let info_b = ks
            .add_factor(&kek, &Factor::RecoveryCode(&b), Some("drawer"), NOW)
            .unwrap();
        assert_ne!(info_a.id, info_b.id);
        assert_eq!(*ks.unseal_with_recovery_code(&a).unwrap(), *kek);
        assert_eq!(*ks.unseal_with_recovery_code(&b).unwrap(), *kek);
        assert_eq!(
            ks.identify(&Factor::RecoveryCode(&b))
                .unwrap()
                .label
                .as_deref(),
            Some("drawer")
        );

        let generation = ks.generation;
        assert_eq!(ks.revoke_factor(&kek, &info_a.id).unwrap(), info_a);
        assert_eq!(ks.generation, generation + 1);
        ks.verify_integrity(&kek).unwrap();
        assert!(ks.unseal_with_recovery_code(&a).is_err(), "revoked");
        assert_eq!(*ks.unseal_with_recovery_code(&b).unwrap(), *kek);
    }

    #[test]
    fn a_code_typed_back_from_a_kit_opens() {
        let (mut ks, kek) = fresh("pw");
        let code = generate_recovery_code();
        ks.add_factor(&kek, &Factor::RecoveryCode(&code), None, NOW)
            .unwrap();
        let typed = crate::keyring::format_code(&code.to_ascii_uppercase()).replace('-', " - ");
        assert_eq!(*ks.unseal_with_recovery_code(&typed).unwrap(), *kek);
    }

    #[test]
    fn the_passphrase_cannot_be_revoked_and_unknown_ids_are_refused() {
        let (mut ks, kek) = fresh("pw");
        assert!(ks.revoke_factor(&kek, PROVIDER_PASSPHRASE).is_err());
        assert!(ks.revoke_factor(&kek, "recovery-00000000").is_err());
        assert_eq!(ks.generation, 0, "a refused revocation writes nothing");
    }

    #[test]
    fn a_factor_is_never_added_under_a_foreign_master_key() {
        let (mut ks, _) = fresh("pw");
        let code = generate_recovery_code();
        let foreign = Zeroizing::new([3u8; KEY_LEN]);
        assert!(ks
            .add_factor(&foreign, &Factor::RecoveryCode(&code), None, NOW)
            .is_err());
        assert_eq!(ks.factors().len(), 1);
    }

    #[test]
    fn a_device_secret_opens_only_its_own_wrap() {
        let (mut ks, kek) = fresh("pw");
        let secret = [7u8; 32];
        let passkey = |id, secret| Factor::Secret {
            provider: PROVIDER_PASSKEY,
            id,
            secret,
        };
        ks.add_factor(&kek, &passkey("cred-a", &secret), Some("YubiKey"), NOW)
            .unwrap();
        assert_eq!(*ks.unseal(&passkey("cred-a", &secret)).unwrap(), *kek);
        assert!(ks.unseal(&passkey("cred-a", &[8u8; 32])).is_err());
        assert!(ks.unseal(&passkey("cred-b", &secret)).is_err());
        let as_enclave = Factor::Secret {
            provider: PROVIDER_SECURE_ENCLAVE,
            id: "cred-a",
            secret: &secret,
        };
        assert!(ks.unseal(&as_enclave).is_err(), "bound to its provider");
        assert!(ks
            .add_factor(&kek, &passkey("short", &[1u8; 16]), None, NOW)
            .is_err());

        ks.add_factor(&kek, &passkey("cred-a", &[9u8; 32]), None, NOW)
            .unwrap();
        assert_eq!(
            ks.factors()
                .iter()
                .filter(|f| f.provider == PROVIDER_PASSKEY)
                .count(),
            1,
            "re-enrolling an id replaces its wrap"
        );
    }

    #[test]
    fn rotating_the_master_key_keeps_records_purpose_keys_blinder_and_ring() {
        let (mut ks, kek) = fresh("pw");
        let code = generate_recovery_code();
        ks.add_factor(&kek, &Factor::RecoveryCode(&code), None, NOW)
            .unwrap();
        ks.add_factor(
            &kek,
            &Factor::Secret {
                provider: PROVIDER_SECURE_ENCLAVE,
                id: "se-1",
                secret: &[5u8; 32],
            },
            Some("Touch ID"),
            NOW,
        )
        .unwrap();
        let record = RecordCryptors::from_keystore(&ks, &kek)
            .unwrap()
            .active()
            .encrypt_record(b"a memory", b"m-1")
            .unwrap();
        let purpose = derive_secret_key(&ks.root_key(&kek).unwrap(), "journal");
        let token = IndexBlinder::derive_from_root(&ks.root_key(&kek).unwrap()).blind("term");
        let ring = ks.ring_key(&kek).unwrap();

        let (new, dropped) = ks.rotate_kek(&kek, &[Factor::Passphrase("pw")]).unwrap();
        assert_ne!(*new, *kek);
        assert!(ks.verify_integrity(&kek).is_err(), "the old key is retired");
        assert_eq!(*ks.unseal_with_passphrase("pw").unwrap(), *new);
        assert!(ks.unseal_with_recovery_code(&code).is_err());
        let dropped: Vec<_> = dropped.iter().map(|f| f.provider.as_str()).collect();
        assert_eq!(dropped, [PROVIDER_RECOVERY, PROVIDER_SECURE_ENCLAVE]);

        let root = ks.root_key(&new).unwrap();
        assert_eq!(
            RecordCryptors::from_keystore(&ks, &new)
                .unwrap()
                .active()
                .decrypt_record(&record, b"m-1")
                .unwrap(),
            b"a memory"
        );
        assert_eq!(*derive_secret_key(&root, "journal"), *purpose);
        assert_eq!(IndexBlinder::derive_from_root(&root).blind("term"), token);
        assert_eq!(ks.ring_key(&new).unwrap().0, ring.0);

        let (newer, _) = ks.rotate_kek(&new, &[Factor::Passphrase("pw")]).unwrap();
        assert_eq!(
            *ks.root_key(&newer).unwrap(),
            *root,
            "a second rotation keeps the same root"
        );
    }

    #[test]
    fn a_rotation_that_would_leave_no_way_in_changes_nothing() {
        let (mut ks, kek) = fresh("pw");
        let before = ks.to_json().unwrap();
        let code = generate_recovery_code();
        assert!(ks.rotate_kek(&kek, &[Factor::RecoveryCode(&code)]).is_err());
        assert_eq!(ks.to_json().unwrap(), before);
    }

    #[test]
    fn a_keystore_without_the_new_fields_keeps_its_bytes_and_its_mac() {
        let (ks, kek) = fresh("pw");
        let json = ks.to_json().unwrap();
        for field in ["root_wrap", "\"id\"", "\"label\"", "created_at"] {
            assert!(
                !json.contains(field),
                "{field} written for a keystore that has none"
            );
        }
        let reloaded = Keystore::from_json(&json).unwrap();
        assert_eq!(reloaded.to_json().unwrap(), json);
        reloaded.verify_integrity(&kek).unwrap();
    }

    #[test]
    fn a_keystore_created_with_a_device_factor_opens_only_through_it() {
        let factor = Factor::Secret {
            provider: PROVIDER_RING,
            id: "ring-0011",
            secret: &[4u8; 32],
        };
        let ks = Keystore::create_with_factor(&factor, NOW).unwrap();
        let kek = ks.unseal(&factor).unwrap();
        ks.verify_integrity(&kek).unwrap();
        assert!(ks.has_ring("ring-0011"));
        assert!(ks.unseal_with_passphrase("anything").is_err());
    }

    #[test]
    fn superseding_save_leaves_no_backup_that_opens_under_a_revoked_code() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keystore.json");
        let (mut ks, kek) = fresh("pw");
        let code = generate_recovery_code();
        let info = ks
            .add_factor(&kek, &Factor::RecoveryCode(&code), None, NOW)
            .unwrap();
        ks.save_to_path(&path).unwrap();
        ks.revoke_factor(&kek, &info.id).unwrap();
        ks.save_superseding(&path).unwrap();
        let bak = std::fs::read_to_string(path.with_extension("json.bak")).unwrap();
        assert_eq!(bak, std::fs::read_to_string(&path).unwrap());
        assert!(Keystore::from_json(&bak)
            .unwrap()
            .unseal_with_recovery_code(&code)
            .is_err());
    }
}
