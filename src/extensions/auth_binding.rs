//! API key to user identity binding for multi-tenant deployments.

use std::collections::HashMap;
use std::path::Path;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::auth::{validate_api_key, AuthError};

pub type KeyHash = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyBinding {
    pub key_hash: KeyHash,
    pub user_id: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub label: Option<String>,
}

pub struct KeyUserBindings {
    bindings: RwLock<HashMap<KeyHash, KeyBinding>>,
    path: std::path::PathBuf,
}

impl KeyUserBindings {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let bindings = if path.exists() {
            let data = std::fs::read_to_string(&path)?;
            // An empty file is a legitimate "no bindings yet" state; only a
            // non-empty file that fails to parse is treated as corrupt. A corrupt
            // file must NOT silently degrade to empty bindings — that would
            // disable tenant isolation (every key would run unbound).
            let records: Vec<KeyBinding> = if data.trim().is_empty() {
                Vec::new()
            } else {
                serde_json::from_str(&data).map_err(|error| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "key-user bindings file {} is corrupt: {error}. Refusing \
                             to continue (empty bindings would disable tenant isolation).",
                            path.display()
                        ),
                    )
                })?
            };
            records
                .into_iter()
                .map(|binding| (binding.key_hash.clone(), binding))
                .collect()
        } else {
            HashMap::new()
        };

        Ok(Self {
            bindings: RwLock::new(bindings),
            path,
        })
    }

    pub fn register(
        &self,
        plaintext_key: &str,
        user_id: &str,
        label: Option<&str>,
    ) -> std::io::Result<()> {
        let binding = KeyBinding {
            key_hash: hash_api_key(plaintext_key),
            user_id: user_id.to_string(),
            created_at: chrono::Utc::now(),
            label: label.map(String::from),
        };

        {
            let mut map = self.bindings.write();
            map.insert(binding.key_hash.clone(), binding);
        }

        self.persist()
    }

    pub fn lookup_user(&self, plaintext_key: &str) -> Option<String> {
        let hash = hash_api_key(plaintext_key);
        self.bindings.read().get(&hash).map(|binding| binding.user_id.clone())
    }

    fn persist(&self) -> std::io::Result<()> {
        let records: Vec<KeyBinding> = self.bindings.read().values().cloned().collect();
        let json = serde_json::to_string_pretty(&records)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

        let tmp_path = self.path.with_extension("json.tmp");
        std::fs::write(&tmp_path, &json)?;
        std::fs::rename(&tmp_path, &self.path)?;
        Ok(())
    }
}

pub fn hash_api_key(plaintext_key: &str) -> KeyHash {
    let mut hasher = Sha256::new();
    hasher.update(plaintext_key.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub fn validate_api_key_with_user(
    plaintext_key: &str,
    bindings: &KeyUserBindings,
) -> Result<Option<String>, AuthError> {
    validate_api_key(plaintext_key)?;
    // Multi-tenant: a valid key with no binding is a misconfiguration, not a
    // superuser. Reject it (fail-closed) so it cannot impersonate any user_id.
    match bindings.lookup_user(plaintext_key) {
        Some(user_id) => Ok(Some(user_id)),
        None => Err(AuthError::InvalidApiKey),
    }
}