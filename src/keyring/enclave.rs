//! Touch ID as a keyring factor, through the `veld-enclave` helper (macOS).
//!
//! The helper holds a Secure Enclave key and releases a 32-byte secret only after
//! Touch ID; see `packaging/macos/veld-enclave`. The keyring's master key is wrapped
//! under that secret as a `secure-enclave` factor with the helper's id.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use serde::Deserialize;
use zeroize::Zeroizing;

pub const HELPER_ENV: &str = "VELD_ENCLAVE_HELPER";
const HELPER: &str = "veld-enclave";

/// `VELD_ENCLAVE_HELPER`, else `veld-enclave` beside this executable, else on PATH.
pub fn helper_path() -> PathBuf {
    if let Ok(path) = std::env::var(HELPER_ENV) {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(HELPER)))
        .filter(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from(HELPER))
}

#[derive(Deserialize)]
struct Released {
    id: String,
    label: Option<String>,
    secret: String,
}

/// A secret the Secure Enclave released after Touch ID.
pub struct EnclaveSecret {
    pub id: String,
    pub label: Option<String>,
    pub secret: Zeroizing<Vec<u8>>,
}

fn run(args: &[&str]) -> Result<EnclaveSecret> {
    let helper = helper_path();
    let out = Command::new(&helper)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("run {helper:?}; build it from packaging/macos/veld-enclave"))?;
    if !out.status.success() {
        bail!(
            "{} failed: {}",
            helper.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let stdout = Zeroizing::new(out.stdout);
    let released: Released = serde_json::from_slice(&stdout).with_context(|| {
        format!(
            "{} printed something other than its report",
            helper.display()
        )
    })?;
    let secret = Zeroizing::new(
        base64::engine::general_purpose::STANDARD
            .decode(released.secret.as_bytes())
            .map_err(|_| anyhow!("{} released a malformed secret", helper.display()))?,
    );
    if secret.len() != 32 {
        bail!(
            "{} released {} bytes, not 32",
            helper.display(),
            secret.len()
        );
    }
    Ok(EnclaveSecret {
        id: released.id,
        label: released.label,
        secret,
    })
}

/// Create a Secure Enclave key, confirmed with Touch ID, and release its secret.
pub fn enrol(label: Option<&str>, policy: &str) -> Result<EnclaveSecret> {
    let mut args = vec!["enrol", "--policy", policy];
    if let Some(label) = label {
        args.extend(["--label", label]);
    }
    run(&args)
}

/// Release the secret of the Secure Enclave key `id`; prompts for Touch ID.
pub fn unlock(id: &str, reason: &str) -> Result<EnclaveSecret> {
    let released = run(&["unlock", "--id", id, "--reason", reason])?;
    if released.id != id {
        bail!("asked the helper for {id} and it released {}", released.id);
    }
    Ok(released)
}
