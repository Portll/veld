//! `veld keys`: one keyring holds the factors, and every tenant keystore opens through
//! it, including a tenant created after the factors were enrolled.

use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;
use veld::keystore::{open_record_keys, Keystore};

const PASS: &str = "keyring-cli-correct-horse-battery-K1";
const NEW_PASS: &str = "keyring-cli-new-passphrase-K2";

fn keys(storage: &Path, args: &[&str], env: &[(&str, &str)]) -> serde_json::Value {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_veld"));
    cmd.arg("keys")
        .args(args)
        .arg("--storage")
        .arg(storage)
        .env_remove("VELD_MASTER_PASSPHRASE")
        .env_remove("VELD_NEW_MASTER_PASSPHRASE")
        .env_remove("VELD_RECOVERY_CODE")
        .env_remove("VELD_KEYSTORE_PATH")
        .env_remove("VELD_KEYRING_PATH")
        .env("VELD_NOW", "2026-10-08T00:00:00Z");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run veld");
    assert!(
        out.status.success(),
        "veld keys {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("JSON report")
}

fn keys_fails(storage: &Path, args: &[&str], env: &[(&str, &str)]) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_veld"));
    cmd.arg("keys")
        .args(args)
        .arg("--storage")
        .arg(storage)
        .env_remove("VELD_MASTER_PASSPHRASE")
        .env_remove("VELD_KEYSTORE_PATH")
        .env_remove("VELD_KEYRING_PATH");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run veld");
    assert!(!out.status.success(), "veld keys {args:?} should refuse");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn store_keystore(storage: &Path, tenant: &str) -> PathBuf {
    storage.join(tenant).join("storage").join("keystore.json")
}

fn load(path: &Path) -> Keystore {
    Keystore::from_json(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn code_from_kit(path: &Path) -> String {
    let kit = std::fs::read_to_string(path).unwrap();
    let grouped = kit
        .lines()
        .skip_while(|l| l.trim() != "Recovery code")
        .find(|l| l.trim().contains('-'))
        .expect("a code in the kit")
        .trim()
        .to_string();
    assert_eq!(grouped.replace('-', "").len(), 48);
    grouped
}

#[test]
fn one_keyring_recovers_rotates_and_opens_every_tenant() {
    std::env::remove_var("VELD_KEYSTORE_PATH");
    std::env::remove_var("VELD_KEYRING_PATH");
    std::env::remove_var("VELD_KMS_WRAP_KEY");
    let data = TempDir::new().unwrap();
    let storage = data.path();

    std::env::set_var("VELD_MASTER_PASSPHRASE", PASS);
    let alice_record = {
        let keys = open_record_keys(&store_keystore(storage, "alice"))
            .unwrap()
            .unwrap();
        keys.cryptors
            .active()
            .encrypt_record(b"alice's memory", b"m-1")
            .unwrap()
    };
    let alice_token = open_record_keys(&store_keystore(storage, "alice"))
        .unwrap()
        .unwrap()
        .blinder
        .blind("term");
    open_record_keys(&store_keystore(storage, "bob"))
        .unwrap()
        .unwrap();
    std::env::remove_var("VELD_MASTER_PASSPHRASE");

    // init: a keyring, both tenants enrolled, a kit written 0600.
    let kit = storage.join("kit-1.txt");
    let init = keys(
        storage,
        &["init", "--kit", kit.to_str().unwrap()],
        &[("VELD_MASTER_PASSPHRASE", PASS)],
    );
    assert_eq!(init["enrolled"].as_array().unwrap().len(), 2);
    assert!(
        init["recovery"].get("code").is_none(),
        "a kit file keeps the code off stdout"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&kit).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    let ring = init["ring"].as_str().unwrap().to_string();
    for tenant in ["alice", "bob"] {
        assert!(load(&store_keystore(storage, tenant)).has_ring(&ring));
    }
    let status = keys(storage, &["status"], &[]);
    let providers: Vec<_> = status["keyring"]["factors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["provider"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(providers, ["passphrase", "recovery"]);
    assert!(
        keys_fails(storage, &["init"], &[("VELD_MASTER_PASSPHRASE", PASS)])
            .contains("already exists")
    );

    // recover: the kit's code re-seals everything under a new passphrase and is replaced.
    let code = code_from_kit(&kit);
    let kit2 = storage.join("kit-2.txt");
    let recovered = keys(
        storage,
        &["recover", "--kit", kit2.to_str().unwrap()],
        &[
            ("VELD_RECOVERY_CODE", &code),
            ("VELD_NEW_MASTER_PASSPHRASE", NEW_PASS),
        ],
    );
    assert_eq!(recovered["keystores"].as_array().unwrap().len(), 2);
    for tenant in ["alice", "bob"] {
        let ks = load(&store_keystore(storage, tenant));
        assert!(ks.unseal_with_passphrase(PASS).is_err());
        ks.unseal_with_passphrase(NEW_PASS).unwrap();
    }
    let keyring = load(&storage.join("keyring.json"));
    assert!(
        keyring.unseal_with_recovery_code(&code).is_err(),
        "the used code is revoked"
    );
    let code2 = code_from_kit(&kit2);
    keyring.unseal_with_recovery_code(&code2).unwrap();
    assert!(
        load(&storage.join("keyring.json.bak"))
            .unseal_with_recovery_code(&code)
            .is_err(),
        "nor does the backup open under it"
    );

    // The server: the keyring opens an enrolled tenant, and a new tenant joins it, with
    // no passphrase left in the environment once the keyring is unsealed.
    std::env::set_var("VELD_MASTER_PASSPHRASE", NEW_PASS);
    assert_eq!(
        veld::keyring::install(storage).unwrap().as_deref(),
        Some(ring.as_str())
    );
    assert!(
        veld::keyring::installed_cryptor("totp").is_some(),
        "TOTP secrets can seal under the keyring when no global keystore is set"
    );
    std::env::remove_var("VELD_MASTER_PASSPHRASE");
    let alice = open_record_keys(&store_keystore(storage, "alice"))
        .unwrap()
        .unwrap();
    assert_eq!(
        alice
            .cryptors
            .active()
            .decrypt_record(&alice_record, b"m-1")
            .unwrap(),
        b"alice's memory"
    );
    open_record_keys(&store_keystore(storage, "carol"))
        .unwrap()
        .expect("a new tenant is created under the keyring");
    assert!(load(&store_keystore(storage, "carol")).has_ring(&ring));

    // rotate-master-keys: new master keys everywhere; data, blinder and ring carry over.
    let kit3 = storage.join("kit-3.txt");
    let rotated = keys(
        storage,
        &["rotate-master-keys", "--kit", kit3.to_str().unwrap()],
        &[("VELD_MASTER_PASSPHRASE", NEW_PASS)],
    );
    assert_eq!(rotated["keystores"].as_array().unwrap().len(), 3);
    assert!(load(&storage.join("keyring.json"))
        .unseal_with_recovery_code(&code2)
        .is_err());
    let alice_after = open_record_keys(&store_keystore(storage, "alice"))
        .unwrap()
        .unwrap();
    assert_eq!(
        alice_after
            .cryptors
            .active()
            .decrypt_record(&alice_record, b"m-1")
            .unwrap(),
        b"alice's memory"
    );
    assert_eq!(alice_after.blinder.blind("term"), alice_token);
    assert!(load(&store_keystore(storage, "carol")).has_ring(&ring));

    // rotate-data-keys with the new recovery code, then revoke that code.
    let code3 = code_from_kit(&kit3);
    let epochs = keys(
        storage,
        &["rotate-data-keys", "--with-recovery-code"],
        &[("VELD_RECOVERY_CODE", &code3)],
    );
    assert!(epochs.as_array().unwrap().iter().all(|e| e["epoch"] == 1));
    let id = rotated["recovery"]["id"].as_str().unwrap();
    keys(
        storage,
        &["revoke", id],
        &[("VELD_MASTER_PASSPHRASE", NEW_PASS)],
    );
    assert!(load(&storage.join("keyring.json"))
        .unseal_with_recovery_code(&code3)
        .is_err());
    assert!(keys_fails(
        storage,
        &["revoke", "passphrase"],
        &[("VELD_MASTER_PASSPHRASE", NEW_PASS)]
    )
    .contains("cannot be revoked"));
}

/// A stand-in for `veld-enclave` that releases a fixed secret, or another one when
/// told to, as Touch ID on a different key would.
fn fake_enclave(dir: &Path, secret: &[u8; 32]) -> PathBuf {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(secret);
    let path = dir.join("veld-enclave");
    std::fs::write(
        &path,
        format!(
            "#!/bin/sh\ncase \"$1\" in\n  enrol|unlock) printf '{{\"id\":\"se-00112233aabbccdd\",\"label\":\"Test Touch ID\",\"secret\":\"%s\"}}\\n' '{b64}' ;;\n  *) echo \"unexpected $1\" >&2; exit 1 ;;\nesac\n"
        ),
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    path
}

#[test]
fn touch_id_enrolled_through_the_helper_recovers_a_lost_passphrase() {
    let data = TempDir::new().unwrap();
    let storage = data.path();
    keys(
        storage,
        &["init", "--kit", storage.join("kit.txt").to_str().unwrap()],
        &[("VELD_MASTER_PASSPHRASE", PASS)],
    );
    let helper = fake_enclave(storage, &[7u8; 32]);
    let enrolled = keys(
        storage,
        &["enrol-touch-id", "--label", "MacBook"],
        &[
            ("VELD_MASTER_PASSPHRASE", PASS),
            ("VELD_ENCLAVE_HELPER", helper.to_str().unwrap()),
        ],
    );
    assert_eq!(enrolled["provider"], "secure-enclave");
    assert_eq!(enrolled["id"], "se-00112233aabbccdd");

    let imposter = TempDir::new().unwrap();
    let wrong = fake_enclave(imposter.path(), &[8u8; 32]);
    keys_fails(
        storage,
        &["recover", "--touch-id", "se-00112233aabbccdd"],
        &[
            ("VELD_ENCLAVE_HELPER", wrong.to_str().unwrap()),
            ("VELD_NEW_MASTER_PASSPHRASE", NEW_PASS),
        ],
    );
    assert!(load(&storage.join("keyring.json"))
        .unseal_with_passphrase(PASS)
        .is_ok());

    let recovered = keys(
        storage,
        &["recover", "--touch-id", "se-00112233aabbccdd"],
        &[
            ("VELD_ENCLAVE_HELPER", helper.to_str().unwrap()),
            ("VELD_NEW_MASTER_PASSPHRASE", NEW_PASS),
        ],
    );
    assert_eq!(recovered["used"]["provider"], "secure-enclave");
    assert!(
        recovered["reissued"].is_null(),
        "a device factor stays enrolled"
    );
    let keyring = load(&storage.join("keyring.json"));
    assert!(keyring.unseal_with_passphrase(PASS).is_err());
    keyring.unseal_with_passphrase(NEW_PASS).unwrap();
}
