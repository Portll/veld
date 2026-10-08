# Key management

How the keys that seal veld's data are set up, recovered and rotated. What the
encryption covers is in [ENCRYPTION-SECURITY.md](ENCRYPTION-SECURITY.md).

## The keys

Every sealed store has a keystore file holding a random **master key** (KEK). The
master key wraps the **data keys** that seal records, one per epoch. It also yields
the **purpose keys** (journal, slow store, TOTP secrets) and the **index blinder**.
A keystore stores the master key only in wrapped form, once per **factor** that can
open it:

| Factor | Opens with | Notes |
|---|---|---|
| `passphrase` | `VELD_MASTER_PASSPHRASE` | How the server unseals at start. Rotated, never revoked. |
| `recovery` | a recovery code | 192 bits, issued once in a recovery kit. Any number; each revocable. |
| `kms` | `VELD_KMS_WRAP_KEY` | Unattended start without a passphrase. |
| `secure-enclave` | Touch ID on one Mac | A Secure Enclave key, through the `veld-enclave` helper. |
| `passkey` | a passkey's PRF secret | Touch ID, Face ID or a security key, through a browser. |
| `ring` | the keyring | Present on every keystore enrolled in a keyring. |

## The keyring

Each tenant keeps its own keystore and master key, so tenants stay cryptographically
separate. Enrolling a recovery code or a passkey in every one of them would break
for the next tenant to appear. The **keyring** (`<data dir>/keyring.json`, or
`VELD_KEYRING_PATH`) avoids this. It is a keystore holding the factors, and every
other keystore carries a `ring` wrap that the keyring opens. A factor enrolled once
on the keyring reaches every tenant, including tenants created later.

At start the server unseals the keyring with the passphrase or the KMS key. It then
opens each tenant through its ring wrap, with no further Argon2 run per tenant. A
tenant that has no ring wrap yet is opened with the passphrase and gains one. A new
tenant is created with a passphrase wrap if a passphrase is set, plus a ring wrap.

Without a keyring, every keystore stands alone as before.

## Set up

```sh
export VELD_MASTER_PASSPHRASE=…          # the passphrase the server already uses
veld keys init --kit ~/veld-recovery-kit.txt
```

`init` does three things:

- creates the keyring under the passphrase;
- enrols every existing keystore in it;
- writes the first recovery kit, which is created with mode 0600 and never overwrites an existing file.

Without `--kit`, the code is printed once in the JSON report instead. Store the kit
offline and apart from any backup of the data directory. `veld keys status` lists
every keystore and its factors without unsealing anything.

## Recover a lost passphrase

```sh
veld keys recover --kit ~/veld-recovery-kit-2.txt
```

It asks for the recovery code and a new passphrase, or reads `VELD_RECOVERY_CODE` and
`VELD_NEW_MASTER_PASSPHRASE`. It opens the keyring with the code, then re-seals the
keyring and every keystore under the new passphrase. The used code is revoked and a
replacement kit is written. Set the new passphrase where the server reads it before
the next start.

Recover with Touch ID or a passkey instead of a code with `veld keys recover --touch-id <id>` or
`veld keys recover --passkey`. A device factor stays enrolled after it recovers.

## Touch ID and passkeys

Both are device factors: something you hold releases a 32-byte secret, and the keyring's
master key is wrapped under a key derived from it (HKDF-SHA256, bound to the factor's
kind). Every `veld keys` command that opens the keyring takes `--with-touch-id <id>`,
`--with-passkey` or `--with-recovery-code` instead of the passphrase.

**Touch ID** (macOS). Build the helper once and put it beside `veld` or on `PATH`, or
point `VELD_ENCLAVE_HELPER` at it:

```sh
swift build -c release --package-path packaging/macos/veld-enclave
veld keys enrol-touch-id --label "MacBook"
```

The helper creates a P-256 key inside the Secure Enclave that only the current
fingerprints can use (`--policy presence` also accepts the login password). It agrees a
secret with that key through an ephemeral key, asks for Touch ID once to prove the secret
comes back, and keeps a handle in `~/Library/Application Support/veld/enclave/`. The
private key never leaves the Secure Enclave, so the factor works on that Mac only, and
adding or removing a fingerprint invalidates it. Keep a recovery code as well.

**Passkeys.** `veld keys enrol-passkey` opens a page on `http://localhost:<port>` in the
browser. It creates a passkey with the WebAuthn PRF extension and posts the PRF secret
back. The relying party is `localhost`, so the same passkey unlocks a keyring on this
machine when no server can start. Passkeys from iCloud Keychain (macOS 15 and later) and
most FIDO2 security keys support PRF; an authenticator without it is refused at
enrolment.

## The Monitor's Security tab and the HTTP API

Veld Monitor's 🔐 Security tab covers the same operations against the instance in the
active tab. Signing in uses user auth (`VELD_USER_AUTH_ENABLED=true`, see
[user-auth.md](user-auth.md)). Key management needs an admin account with 2FA. Each change
asks for a 2FA code (one per five minutes per session) and for something that opens the
keyring: the passphrase, a recovery code, or Touch ID on the Mac running the Monitor.
Recovery kits are saved where you choose, readable by you only. A passkey added from the
Monitor is created in the system browser on the server's own origin, so its relying party
is the server's host name.

The routes are under `/api/user_auth/admin/keys/`: `status`, `init`, `recovery_codes`,
`revoke`, `rotate_passphrase`, `rotate_data_keys`, `rotate_master_keys`, `factors` and
`passkey`. A change carries the unlock factor in its body as one of
`{"passphrase": …}`, `{"recovery_code": …}`, `{"secure_enclave": {"id", "secret"}}` or
`{"passkey": {"id", "secret"}}`. Failed unlocks share one budget across the server. On a
non-loopback bind the routes refuse unless `VELD_TLS_ACK=true` says TLS terminates in
front of veld, because the bodies carry passphrases. Every change is in the audit trail.

## Rotate

| Command | What changes | Use it when |
|---|---|---|
| `veld keys rotate-passphrase` | The passphrase wrap of the keyring and of every keystore. | The passphrase may be known to someone it should not be. |
| `veld keys add-recovery-code`, `veld keys revoke <id>` | One factor. | A code is lost, or a device is retired. |
| `veld keys rotate-data-keys` | A new data-key epoch in every keystore. New writes seal under it, and earlier epochs stay readable. | Routine rotation, and limiting how much data one data key seals. |
| `veld keys rotate-master-keys` | A new master key everywhere. Data keys, purpose keys, the index blinder and ring wraps are re-wrapped and keep working. Every recovery code and device factor is dropped, and a new kit is issued. | A master key may have been exposed, or a clean start for every factor is wanted. |

A revocation or master-key rotation also overwrites `keystore.json.bak`, because the
previous file would still open under what was removed. Copies elsewhere still open
under the factors they held: backups, snapshots, another disk. To make those copies
useless, rotate the data keys and re-seal existing records under the new epoch.
Re-sealing existing records is not automated yet.

## Files

- `keyring.json`: the keyring. Losing it loses the recovery codes and device factors
  (the passphrase still opens each keystore).
- `keystore.json.lock`: advisory lock taken by every writer of a keystore. Safe to
  delete while nothing is running.
- `keystore.json.bak`: the previous version of a keystore. Restoring it needs
  `VELD_ALLOW_KEYSTORE_ROLLBACK=true` once, because the store's rollback sentinel
  refuses an older generation.
