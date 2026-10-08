//! The keyring: one keystore holding every factor an operator unlocks with, which
//! opens every other keystore on the server.
//!
//! Each tenant's store keystore, and the global one at VELD_KEYSTORE_PATH, keeps its
//! own master key, so tenants stay cryptographically separate. Beside its passphrase
//! wrap each carries a `ring` wrap: its master key sealed under a key derived from the
//! keyring. Factors are enrolled once, on the keyring. A recovery code, a passkey or a
//! Secure Enclave key opens the keyring, and the keyring opens every tenant, including
//! tenants created after the factor was enrolled.
//!
//! The ring key derives from the keyring's root rather than its master key, so
//! rotating the keyring's master key leaves every tenant's ring wrap valid.
//!
//! Without a keyring each keystore stands alone, as before, unsealed with the
//! passphrase or VELD_KMS_WRAP_KEY.
//!
//! Every write takes `<file>.lock` and every multi-keystore operation is computed in
//! memory before the first file is replaced, so a refused operation changes nothing.

pub mod enclave;
pub mod passkey;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use zeroize::Zeroizing;

use crate::config::StorageBackend;
use crate::keystore::{
    store_keystore_path, Factor, FactorInfo, KdfParams, Keystore, LocalAeadKms, SecretCryptor,
    SecretKey, PROVIDER_PASSPHRASE, PROVIDER_RECOVERY, PROVIDER_RING,
};

pub const KEYRING_FILE: &str = "keyring.json";
pub const KEYRING_PATH_ENV: &str = "VELD_KEYRING_PATH";

pub fn keyring_path(storage: &Path) -> PathBuf {
    match std::env::var(KEYRING_PATH_ENV) {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p),
        _ => storage.join(KEYRING_FILE),
    }
}

/// What a keyring lends the keystores it opens: the key their `ring` wraps are sealed
/// under and the id those wraps carry.
#[derive(Clone)]
pub struct RingKey {
    id: String,
    secret: SecretKey,
}

impl RingKey {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn factor(&self) -> Factor<'_> {
        Factor::Secret {
            provider: PROVIDER_RING,
            id: &self.id,
            secret: self.secret.as_slice(),
        }
    }
}

/// What the keyring unsealed at start leaves in memory: the ring key, and the
/// purpose keys of server-wide secrets sealed under the keyring.
#[derive(Clone)]
struct Installed {
    ring: RingKey,
    totp: SecretCryptor,
}

static INSTALLED: RwLock<Option<Installed>> = RwLock::new(None);

/// The ring key unsealed at start, if this process has a keyring.
pub fn installed() -> Option<RingKey> {
    INSTALLED
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .as_ref()
        .map(|i| i.ring.clone())
}

/// A server-wide purpose key from the keyring unsealed at start. TOTP secrets are
/// sealed under it when no VELD_KEYSTORE_PATH keystore is set.
pub fn installed_cryptor(purpose: &str) -> Option<SecretCryptor> {
    let installed = INSTALLED
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match purpose {
        "totp" => installed.as_ref().map(|i| i.totp.clone()),
        _ => None,
    }
}

/// Unseal the keyring unattended so tenant keystores open through it. `Ok(None)` when
/// there is no keyring. A keyring present but not unsealable is an error: starting
/// without it would mean tenants it alone can open are refused one at a time.
pub fn install(storage: &Path) -> Result<Option<String>> {
    let path = keyring_path(storage);
    let installed = match read_keystore(&path)? {
        None => None,
        Some(ks) => {
            let kek = unseal_unattended(&ks, &path)?;
            ks.verify_integrity(&kek)
                .with_context(|| format!("{path:?} failed its integrity check"))?;
            Some(Installed {
                ring: ring_key_of(&ks, &kek)?,
                totp: ks.purpose_cryptor(&kek, "keyring:totp")?,
            })
        }
    };
    let id = installed.as_ref().map(|i| i.ring.id.clone());
    *INSTALLED
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = installed;
    if let Some(id) = &id {
        tracing::info!(ring = %id, "keyring unsealed: tenant keystores open through it");
    }
    Ok(id)
}

fn unseal_unattended(ks: &Keystore, path: &Path) -> Result<SecretKey> {
    if let Some(pass) = std::env::var("VELD_MASTER_PASSPHRASE")
        .ok()
        .filter(|s| !s.is_empty())
    {
        return ks.unseal(&Factor::Passphrase(&pass));
    }
    if let Some(kms) = LocalAeadKms::from_env()? {
        return ks.unseal(&Factor::Kms(&kms));
    }
    bail!(
        "{path:?} exists but neither VELD_MASTER_PASSPHRASE nor VELD_KMS_WRAP_KEY is set to \
         unseal it"
    )
}

fn ring_key_of(ks: &Keystore, kek: &SecretKey) -> Result<RingKey> {
    let (id, secret) = ks.ring_key(kek)?;
    Ok(RingKey { id, secret })
}

/// Read a keystore file. `Ok(None)` only when it does not exist.
fn read_keystore(path: &Path) -> Result<Option<Keystore>> {
    match std::fs::read_to_string(path) {
        Ok(json) => Keystore::from_json(&json)
            .with_context(|| format!("{path:?}"))
            .map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("read {path:?}")),
    }
}

/// Exclusive advisory lock on `<path>.lock`, held until dropped. Writers of the same
/// keystore, in this process or another, take it before reading what they will write.
pub fn lock_keystore(path: &Path) -> Result<File> {
    let mut name = path.as_os_str().to_owned();
    name.push(".lock");
    let lock_path = PathBuf::from(name);
    if let Some(dir) = lock_path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {dir:?}"))?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("open {lock_path:?}"))?;
    file.lock().with_context(|| format!("lock {lock_path:?}"))?;
    Ok(file)
}

/// A keystore the keyring manages: a tenant's store keystore or the global one.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ManagedKeystore {
    pub scope: String,
    pub path: PathBuf,
}

/// Every keystore under `storage` other than the keyring: the global keystore when
/// VELD_KEYSTORE_PATH names one, then each tenant's store keystore on either backend.
pub fn managed_keystores(storage: &Path) -> Result<Vec<ManagedKeystore>> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("VELD_KEYSTORE_PATH") {
        let path = PathBuf::from(p.trim());
        if !p.trim().is_empty() && path.is_file() {
            out.push(ManagedKeystore {
                scope: "global".into(),
                path,
            });
        }
    }
    let entries = match std::fs::read_dir(storage) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e).with_context(|| format!("list {storage:?}")),
    };
    let mut tenants = Vec::new();
    for entry in entries {
        let entry = entry.with_context(|| format!("list {storage:?}"))?;
        if entry.file_type()?.is_dir() {
            tenants.push(entry.path());
        }
    }
    tenants.sort();
    for dir in tenants {
        let scope = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        for backend in [StorageBackend::RocksDb, StorageBackend::Redb] {
            let path = store_keystore_path(&dir, backend);
            if path.is_file() && !out.iter().any(|m| m.path == path) {
                out.push(ManagedKeystore {
                    scope: scope.clone(),
                    path,
                });
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, Serialize)]
pub struct KeystoreSummary {
    pub scope: String,
    pub path: PathBuf,
    pub generation: Option<u64>,
    pub active_epoch: Option<u32>,
    pub epochs: Vec<u32>,
    pub factors: Vec<FactorInfo>,
    /// The id of this keystore's ring wrap; `None` when the keyring cannot open it.
    pub ring: Option<String>,
    /// Why the file could not be read. A keystore that does not parse is reported,
    /// never left out.
    pub error: Option<String>,
}

fn summarise(scope: &str, path: &Path) -> KeystoreSummary {
    let mut summary = KeystoreSummary {
        scope: scope.to_string(),
        path: path.to_path_buf(),
        generation: None,
        active_epoch: None,
        epochs: Vec::new(),
        factors: Vec::new(),
        ring: None,
        error: None,
    };
    match read_keystore(path) {
        Ok(Some(ks)) => {
            summary.generation = Some(ks.generation);
            summary.active_epoch = Some(ks.active_epoch);
            summary.epochs = ks.deks.iter().map(|d| d.epoch).collect();
            summary.factors = ks.factors();
            summary.ring = summary
                .factors
                .iter()
                .find(|f| f.provider == PROVIDER_RING)
                .map(|f| f.id.clone());
        }
        Ok(None) => summary.error = Some("missing".into()),
        Err(e) => summary.error = Some(format!("{e:#}")),
    }
    summary
}

#[derive(Clone, Debug, Serialize)]
pub struct KeyStatus {
    pub keyring: Option<KeystoreSummary>,
    pub keystores: Vec<KeystoreSummary>,
}

/// What is enrolled where, read without unsealing anything.
pub fn status(storage: &Path) -> Result<KeyStatus> {
    let path = keyring_path(storage);
    let keyring = path.exists().then(|| summarise("keyring", &path));
    let keystores = managed_keystores(storage)?
        .iter()
        .map(|m| summarise(&m.scope, &m.path))
        .collect();
    Ok(KeyStatus { keyring, keystores })
}

/// A recovery code issued once. The code is shown or downloaded once and never stored.
#[derive(Clone, Debug, Serialize)]
pub struct IssuedCode {
    pub id: String,
    pub label: Option<String>,
    pub code: String,
    pub kit: String,
}

/// Group a recovery code for transcription; any grouping is accepted back.
pub fn format_code(code: &str) -> String {
    code.as_bytes()
        .chunks(6)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect::<Vec<_>>()
        .join("-")
}

/// The text of a downloadable recovery kit for `code`.
pub fn recovery_kit(info: &FactorInfo, code: &str, keyring: &Path) -> String {
    let created = info.created_at.as_deref().unwrap_or("unknown");
    let label = info
        .label
        .as_deref()
        .map(|l| format!(" ({l})"))
        .unwrap_or_default();
    format!(
        "VELD RECOVERY KIT\n\
         =================\n\
         \n\
         Created   {created}\n\
         Keyring   {keyring}\n\
         Code id   {id}{label}\n\
         \n\
         Recovery code\n\
         \n    {grouped}\n\
         \n\
         Use it when the passphrase is lost:\n\
         \n    veld keys recover --storage <data directory>\n\
         \n\
         It asks for this code and a new passphrase, re-seals every keystore under the new\n\
         passphrase, revokes this code and issues a replacement.\n\
         \n\
         The code opens the keyring, and the keyring opens every memory on the server. Keep\n\
         this kit offline and apart from any backup of the data directory. Revoke it with:\n\
         \n    veld keys revoke {id}\n",
        keyring = keyring.display(),
        id = info.id,
        grouped = format_code(code),
    )
}

/// The keyring unlocked by one factor and held under its lock.
pub struct Keyring {
    path: PathBuf,
    ks: Keystore,
    kek: SecretKey,
    _lock: File,
}

impl Keyring {
    pub fn open(storage: &Path, factor: &Factor) -> Result<Self> {
        let path = keyring_path(storage);
        let lock = lock_keystore(&path)?;
        let ks = read_keystore(&path)?
            .ok_or_else(|| anyhow!("no keyring at {path:?}; create one with `veld keys init`"))?;
        let kek = ks.unseal(factor)?;
        ks.verify_integrity(&kek)
            .with_context(|| format!("{path:?} failed its integrity check"))?;
        Ok(Self {
            path,
            ks,
            kek,
            _lock: lock,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn keystore(&self) -> &Keystore {
        &self.ks
    }

    pub fn ring_key(&self) -> Result<RingKey> {
        ring_key_of(&self.ks, &self.kek)
    }

    fn issue_code(&mut self, label: Option<&str>, now: &str) -> Result<IssuedCode> {
        let code = Zeroizing::new(crate::keystore::generate_recovery_code());
        let info = self
            .ks
            .add_factor(&self.kek, &Factor::RecoveryCode(&code), label, now)?;
        Ok(IssuedCode {
            kit: recovery_kit(&info, &code, &self.path),
            id: info.id,
            label: info.label,
            code: code.to_string(),
        })
    }
}

/// A managed keystore opened for an operation, with its lock held.
struct OpenTenant {
    managed: ManagedKeystore,
    ks: Keystore,
    kek: SecretKey,
    _lock: File,
}

/// Open every managed keystore through the ring, or with `passphrase` where a keystore
/// has no ring wrap yet; such a keystore gains one. A keystore neither opens refuses
/// the whole operation, naming it.
fn open_tenants(
    storage: &Path,
    ring: &RingKey,
    passphrase: Option<&str>,
) -> Result<Vec<OpenTenant>> {
    let mut open = Vec::new();
    for managed in managed_keystores(storage)? {
        let lock = lock_keystore(&managed.path)?;
        let mut ks =
            read_keystore(&managed.path)?.ok_or_else(|| anyhow!("{:?} vanished", managed.path))?;
        let kek = match ks.unseal(&ring.factor()) {
            Ok(kek) => kek,
            Err(ring_err) => {
                let Some(pass) = passphrase else {
                    bail!(
                        "{} ({:?}) does not open through the keyring ({ring_err:#}); run \
                         `veld keys init` with the passphrase to enrol it",
                        managed.scope,
                        managed.path
                    );
                };
                let kek = ks
                    .unseal(&Factor::Passphrase(pass))
                    .with_context(|| format!("{} ({:?})", managed.scope, managed.path))?;
                ks.verify_integrity(&kek)?;
                ks.add_factor(&kek, &ring.factor(), None, &now_rfc3339())?;
                kek
            }
        };
        ks.verify_integrity(&kek)
            .with_context(|| format!("{} ({:?})", managed.scope, managed.path))?;
        open.push(OpenTenant {
            managed,
            ks,
            kek,
            _lock: lock,
        });
    }
    Ok(open)
}

pub(crate) fn now_rfc3339() -> String {
    std::env::var("VELD_NOW")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
}

/// Replace each file, keystores first and the keyring last, so a failure part-way
/// leaves the keyring able to open what was already written. `superseding` also
/// overwrites each `.bak`: after a revocation the previous file still opens under
/// the revoked factor.
fn persist(keyring: Option<&Keyring>, tenants: &[OpenTenant], superseding: bool) -> Result<()> {
    let save = |ks: &Keystore, path: &Path| {
        if superseding {
            ks.save_superseding(path)
        } else {
            ks.save_to_path(path)
        }
    };
    for t in tenants {
        save(&t.ks, &t.managed.path).with_context(|| format!("write {:?}", t.managed.path))?;
    }
    if let Some(k) = keyring {
        save(&k.ks, &k.path).with_context(|| format!("write {:?}", k.path))?;
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
pub struct InitReport {
    pub keyring: PathBuf,
    pub ring: String,
    pub enrolled: Vec<ManagedKeystore>,
    pub recovery: IssuedCode,
}

/// Create the keyring under `passphrase`, enrol every existing keystore in it and issue
/// the first recovery code. Refuses when a keyring already exists.
pub fn init(storage: &Path, passphrase: &str, label: Option<&str>) -> Result<InitReport> {
    if passphrase.is_empty() {
        bail!("the keyring needs a passphrase; set VELD_MASTER_PASSPHRASE");
    }
    let path = keyring_path(storage);
    let lock = lock_keystore(&path)?;
    if path.exists() {
        bail!("{path:?} already exists");
    }
    let ks = Keystore::create(passphrase, KdfParams::production())?;
    let kek = ks.unseal(&Factor::Passphrase(passphrase))?;
    let mut keyring = Keyring {
        path,
        ks,
        kek,
        _lock: lock,
    };
    let ring = keyring.ring_key()?;
    let tenants = open_tenants(storage, &ring, Some(passphrase))?;
    let recovery = keyring.issue_code(label.or(Some("initial")), &now_rfc3339())?;
    if let Some(dir) = keyring.path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {dir:?}"))?;
    }
    persist(Some(&keyring), &tenants, false)?;
    Ok(InitReport {
        keyring: keyring.path.clone(),
        ring: ring.id,
        enrolled: tenants.into_iter().map(|t| t.managed).collect(),
        recovery,
    })
}

/// Issue another recovery code. Earlier codes stay valid until revoked.
pub fn add_recovery_code(
    storage: &Path,
    unlock: &Factor,
    label: Option<&str>,
) -> Result<IssuedCode> {
    let mut keyring = Keyring::open(storage, unlock)?;
    let issued = keyring.issue_code(label, &now_rfc3339())?;
    persist(Some(&keyring), &[], false)?;
    Ok(issued)
}

/// Enrol a device factor (a passkey's PRF output, a Secure Enclave key's secret) on
/// the keyring. Re-enrolling the same provider and id replaces its wrap.
pub fn enrol_secret(
    storage: &Path,
    unlock: &Factor,
    provider: &str,
    id: &str,
    label: Option<&str>,
    secret: &[u8],
) -> Result<FactorInfo> {
    if [PROVIDER_PASSPHRASE, PROVIDER_RECOVERY, PROVIDER_RING].contains(&provider) {
        bail!("{provider} is not a device factor");
    }
    let mut keyring = Keyring::open(storage, unlock)?;
    let info = keyring.ks.add_factor(
        &keyring.kek,
        &Factor::Secret {
            provider,
            id,
            secret,
        },
        label,
        &now_rfc3339(),
    )?;
    persist(Some(&keyring), &[], false)?;
    Ok(info)
}

/// Remove a factor from the keyring. The passphrase cannot be revoked, only rotated.
pub fn revoke(storage: &Path, unlock: &Factor, id: &str) -> Result<FactorInfo> {
    let mut keyring = Keyring::open(storage, unlock)?;
    let revoked = keyring.ks.revoke_factor(&keyring.kek, id)?;
    persist(Some(&keyring), &[], true)?;
    Ok(revoked)
}

#[derive(Clone, Debug, Serialize)]
pub struct Rewrapped {
    pub keyring: PathBuf,
    pub keystores: Vec<ManagedKeystore>,
}

fn rewrapped(keyring: &Keyring, tenants: Vec<OpenTenant>) -> Rewrapped {
    Rewrapped {
        keyring: keyring.path.clone(),
        keystores: tenants.into_iter().map(|t| t.managed).collect(),
    }
}

/// Re-wrap the keyring and every keystore's passphrase wrap under `new`.
pub fn rotate_passphrase(storage: &Path, old: &str, new: &str) -> Result<Rewrapped> {
    if new.is_empty() || new == old {
        bail!("the new passphrase must be non-empty and differ from the old one");
    }
    let mut keyring = Keyring::open(storage, &Factor::Passphrase(old))?;
    let ring = keyring.ring_key()?;
    let mut tenants = open_tenants(storage, &ring, Some(old))?;
    keyring.ks.set_passphrase(&keyring.kek, new)?;
    for t in &mut tenants {
        t.ks.set_passphrase(&t.kek, new)?;
    }
    persist(Some(&keyring), &tenants, true)?;
    Ok(rewrapped(&keyring, tenants))
}

#[derive(Clone, Debug, Serialize)]
pub struct Recovered {
    pub keyring: PathBuf,
    pub keystores: Vec<ManagedKeystore>,
    /// The factor that recovered, revoked when it was a recovery code.
    pub used: FactorInfo,
    /// The replacement for a used recovery code.
    pub reissued: Option<IssuedCode>,
}

/// Recover from a lost passphrase: open the keyring with a recovery code or device
/// factor and re-seal the keyring and every keystore under `new_passphrase`. A used
/// recovery code is revoked and replaced.
pub fn recover(storage: &Path, unlock: &Factor, new_passphrase: &str) -> Result<Recovered> {
    if matches!(unlock, Factor::Passphrase(_)) {
        bail!("recovery opens the keyring with a recovery code or a device factor");
    }
    if new_passphrase.is_empty() {
        bail!("the new passphrase must be non-empty");
    }
    let mut keyring = Keyring::open(storage, unlock)?;
    let used = keyring.ks.identify(unlock)?;
    let ring = keyring.ring_key()?;
    let mut tenants = open_tenants(storage, &ring, None)?;
    keyring.ks.set_passphrase(&keyring.kek, new_passphrase)?;
    for t in &mut tenants {
        t.ks.set_passphrase(&t.kek, new_passphrase)?;
    }
    let reissued = if used.provider == PROVIDER_RECOVERY {
        keyring.ks.revoke_factor(&keyring.kek, &used.id)?;
        Some(keyring.issue_code(used.label.as_deref(), &now_rfc3339())?)
    } else {
        None
    };
    persist(Some(&keyring), &tenants, true)?;
    Ok(Recovered {
        keyring: keyring.path.clone(),
        keystores: tenants.into_iter().map(|t| t.managed).collect(),
        used,
        reissued,
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct EpochStarted {
    pub scope: String,
    pub path: PathBuf,
    pub epoch: u32,
}

/// Start a new data-key epoch in every keystore. New writes seal under it; records
/// under earlier epochs stay readable with the keys each keystore retains.
pub fn rotate_data_keys(storage: &Path, unlock: &Factor) -> Result<Vec<EpochStarted>> {
    let keyring = Keyring::open(storage, unlock)?;
    let ring = keyring.ring_key()?;
    let passphrase = match unlock {
        Factor::Passphrase(p) => Some(*p),
        _ => None,
    };
    let mut tenants = open_tenants(storage, &ring, passphrase)?;
    let mut epochs = Vec::new();
    for t in &mut tenants {
        epochs.push(t.ks.rotate_dek(&t.kek)?);
    }
    persist(None, &tenants, false)?;
    Ok(tenants
        .into_iter()
        .zip(epochs)
        .map(|(t, epoch)| EpochStarted {
            scope: t.managed.scope,
            path: t.managed.path,
            epoch,
        })
        .collect())
}

#[derive(Clone, Debug, Serialize)]
pub struct MasterKeysRotated {
    pub keyring: PathBuf,
    pub keystores: Vec<ManagedKeystore>,
    /// Factors the old master keys carried that the new ones do not: re-enrol any
    /// still wanted.
    pub dropped: Vec<FactorInfo>,
    pub recovery: IssuedCode,
}

/// Replace the master key of the keyring and of every keystore. Each keeps its data
/// keys, purpose keys and index blinder, re-wrapped under the new master key. Only the
/// passphrase and the ring carry over: every recovery code and device factor is
/// dropped, and a fresh recovery code is issued.
pub fn rotate_master_keys(
    storage: &Path,
    passphrase: &str,
    label: Option<&str>,
) -> Result<MasterKeysRotated> {
    let mut keyring = Keyring::open(storage, &Factor::Passphrase(passphrase))?;
    let ring = keyring.ring_key()?;
    let mut tenants = open_tenants(storage, &ring, Some(passphrase))?;
    let (new_kek, dropped) = keyring
        .ks
        .rotate_kek(&keyring.kek, &[Factor::Passphrase(passphrase)])?;
    keyring.kek = new_kek;
    let ring_after = keyring.ring_key()?;
    if ring_after.id != ring.id {
        bail!("the ring key changed across a master-key rotation; nothing was written");
    }
    let mut tenant_dropped = Vec::new();
    for t in &mut tenants {
        let (kek, dropped) =
            t.ks.rotate_kek(&t.kek, &[Factor::Passphrase(passphrase), ring.factor()])?;
        t.kek = kek;
        tenant_dropped.extend(dropped);
    }
    let recovery = keyring.issue_code(label.or(Some("after rotation")), &now_rfc3339())?;
    persist(Some(&keyring), &tenants, true)?;
    let mut all_dropped = dropped;
    for d in tenant_dropped {
        if !all_dropped.contains(&d) {
            all_dropped.push(d);
        }
    }
    Ok(MasterKeysRotated {
        keyring: keyring.path.clone(),
        keystores: tenants.into_iter().map(|t| t.managed).collect(),
        dropped: all_dropped,
        recovery,
    })
}
