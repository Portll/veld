# User auth — Phase C (password + TOTP + recovery codes)

Veld today is API-key only. Phase C adds a parallel surface for *human*
authentication: a user logs in with a username + password (and, if
enrolled, a TOTP code) and receives an opaque bearer session token to
authorize subsequent calls. This is the substrate the planned Tauri GUI
and TUI git viewer will share.

The entire surface is gated behind the `VELD_USER_AUTH_ENABLED=true`
environment variable. When unset, the RocksDB column family is never
created and the `/api/user_auth/*` routes return 404.

## Environment variables

| Variable                  | Effect                                                                                       |
| ------------------------- | -------------------------------------------------------------------------------------------- |
| `VELD_USER_AUTH_ENABLED`  | `true` / `1` / `yes` / `on` enables the surface. Default: off (no CF; routes answer 503).    |
| `VELD_KEYSTORE_PATH`      | With `VELD_MASTER_PASSPHRASE`, the keystore TOTP secrets are sealed under (purpose `totp`).  |
| `VELD_TLS_ACK`            | Required for key management on a non-loopback bind: TLS terminates in front of veld.        |

Without `VELD_KEYSTORE_PATH`, TOTP secrets are sealed under the keyring when the server
started with one (see [key-management.md](key-management.md)). With neither, 2FA enrollment
is refused in every mode (`TOTP_ENCRYPTION_REQUIRED`): a 2FA seed is never stored in
plaintext.

The login throttle (5 attempts / 15 min, per username) is hard-coded. An admin clears it
for one user with `/api/user_auth/admin/users/unlock`.

## Endpoint shape

All routes live under `/api/user_auth/*` on the protected router. They
are NOT api-key gated — the api-key middleware skips this prefix —
because they're authenticated by a different mechanism. The
session-token middleware is layered on only the routes that require it.

| Method | Path                              | Auth                       | Notes                                                                |
| ------ | --------------------------------- | -------------------------- | -------------------------------------------------------------------- |
| POST   | `/api/user_auth/register`         | Bootstrap-or-Bearer admin  | First call: empty user table → Admin. Otherwise: Bearer admin token. |
| POST   | `/api/user_auth/login`            | Public (creds + TOTP)      | Returns `{session_token, expires_at, role}`. Per-username throttled. |
| POST   | `/api/user_auth/2fa/enroll`       | Bearer session             | Returns provisioning URI + 10 plaintext recovery codes (once).       |
| POST   | `/api/user_auth/2fa/confirm`      | Bearer session             | Finalises enrollment with the first valid TOTP.                      |
| POST   | `/api/user_auth/recover`          | Public (creds + code)      | Consumes one recovery code, resets password, wipes 2FA + sessions.   |
| POST   | `/api/user_auth/logout`           | Bearer session             | Invalidates the bearer.                                              |
| GET    | `/api/user_auth/me`               | Bearer session             | Your account and your sessions.                                      |
| POST   | `/api/user_auth/password`         | Bearer + step-up if 2FA    | `{current_password, new_password}`; ends your other sessions.        |
| POST   | `/api/user_auth/recovery_codes`   | Bearer + step-up           | A fresh batch of 10 login recovery codes, shown once.                |
| POST   | `/api/user_auth/sessions/revoke`  | Bearer session             | `{session_id}`: end one of your own sessions.                        |

Administering logins (Admin session; changes also need step-up):

| Method | Path                                         | Notes                                                       |
| ------ | -------------------------------------------- | ----------------------------------------------------------- |
| GET    | `/api/user_auth/admin/users`                 | Every user: role, 2FA state, lockout, sessions, disabled.   |
| POST   | `/api/user_auth/admin/users/disable`         | `{username}`: cannot sign in; every session ends.           |
| POST   | `/api/user_auth/admin/users/enable`          | `{username}`.                                               |
| POST   | `/api/user_auth/admin/users/reset_password`  | `{username}`: a temporary password, shown once.             |
| POST   | `/api/user_auth/admin/users/reset_2fa`       | `{username}`: clears 2FA and recovery codes.                |
| POST   | `/api/user_auth/admin/users/unlock`          | `{username}`: clears the login lockout.                     |
| GET    | `/api/user_auth/admin/sessions?username=`    | That user's sessions, by public id.                         |
| POST   | `/api/user_auth/admin/sessions/revoke`       | `{username, session_id}`.                                   |
| GET    | `/api/user_auth/admin/audit?limit=`          | Administrative actions, newest first (at most 1000).        |

An admin cannot disable or reset their own account through these routes; the self-service
routes above ask for the current password instead. Key management lives under
`/api/user_auth/admin/keys/*` ([key-management.md](key-management.md)).

### Step-up

A change needs a current TOTP code in the `X-Veld-Step-Up` header. A code is accepted
once per user, so a captured request cannot be replayed inside the code's window. A
successful step-up elevates that one session for five minutes, so a run of changes asks
for one code. Elevation is held in memory: a restart asks again. Wrong codes spend the
caller's own attempt budget. An account without 2FA cannot step up
(`STEP_UP_UNAVAILABLE`): enrol 2FA before administering.

### Wire shapes (JSON)

```jsonc
// register request
{ "username": "alice", "password": "correcthorsebatterystaple" }
// register response (201)
{ "success": true, "user_id": "<uuid>", "role": "admin" | "user" }

// login request
{ "username": "alice", "password": "...", "totp": "123456" }   // totp optional pre-enrollment
// login response (200)
{ "success": true,
  "session_token": "<43-char url-safe base64>",
  "expires_at":   "2026-…",
  "role":         "admin" | "user" }

// 2fa/enroll response (200) — recovery_codes returned ONCE.
{ "success": true,
  "provisioning_uri": "otpauth://totp/Veld:alice?secret=...&issuer=Veld",
  "recovery_codes":   ["ABCDE-FGHJ-KMNPQ-XY", ...] }   // 10 entries

// 2fa/confirm request
{ "totp": "123456" }

// recover request
{ "username": "alice", "recovery_code": "ABCDE-FGHJ-KMNPQ-XY",
  "new_password": "another-strong-passphrase" }
```

Error responses match Veld's standard `ErrorResponse` shape (`code`,
`message`, optional `details` / `request_id`). The codes used are:

| HTTP | `code`                       | When                                                       |
| ---- | ---------------------------- | ---------------------------------------------------------- |
| 401  | `INVALID_CREDENTIALS`        | Wrong password, missing user, wrong TOTP.                  |
| 401  | `TOTP_REQUIRED`              | Account has 2FA, request omitted `totp`.                   |
| 401  | `INVALID_SESSION`            | Bearer is missing, malformed, or expired.                  |
| 401  | `INVALID_RECOVERY_CODE`      | No stored recovery hash matches.                           |
| 400  | `WEAK_PASSWORD`              | < 8 chars or username fails validation.                    |
| 400  | `TOTP_ALREADY_ENROLLED`      | Caller already has active 2FA.                             |
| 400  | `TOTP_NO_PENDING_ENROLLMENT` | `/2fa/confirm` without prior `/2fa/enroll`.                |
| 400  | `TOTP_ENCRYPTION_REQUIRED`   | No keystore or keyring to seal the 2FA secret under.       |
| 401  | `STEP_UP_REQUIRED`           | A change came without a valid `X-Veld-Step-Up` code.       |
| 401  | `STEP_UP_REPLAYED`           | That step-up code was already used.                        |
| 403  | `STEP_UP_UNAVAILABLE`        | The caller has no 2FA to step up with.                     |
| 403  | `ACCOUNT_DISABLED`           | An admin disabled the account.                             |
| 403  | `FORBIDDEN`                  | The route needs the Admin role.                            |
| 409  | `SELF_ADMINISTRATION`        | An admin route aimed at the caller's own account.          |
| 404  | `SESSION_NOT_FOUND`          | No session with that id for that user.                     |
| 409  | `USERNAME_TAKEN`             | Registration: username already maps to a user.             |
| 429  | `TOO_MANY_ATTEMPTS`          | Per-username throttle tripped (5 / 15 min).                |
| 503  | `user_auth_disabled`         | Feature flag is off: every route answers 503.              |

## Sessions

Tokens are 32 random bytes encoded as URL-safe base64 (no padding), 43
chars on the wire. The server only persists the SHA-256 digest, so a
database compromise alone cannot resurrect a live session.

Lifetime: 24 hours, **refreshed on every use**. A session that's exercised
every few minutes effectively never expires; one that's idle for 24h
dies. There is no hard cap. A user ends their own sessions with
`/sessions/revoke` or `/sessions/revoke_mine`; an admin ends one with
`/admin/sessions/revoke`, all of a user's with `/sessions/revoke_all`, and
disabling an account or resetting its password or 2FA ends them all. A
session's public id is the first 16 hex characters of its token hash.

## TOTP

`totp-rs 5.x` with SHA-1 / 30-second / 6-digit profile — the baseline
every authenticator app accepts. Skew tolerance is ±1 step (30 s
either side of "now") to handle mildly mis-synced clocks. The
provisioning URI uses issuer `"Veld"` and the username as the account
label (URL-encoded by `totp-rs`).

The HMAC secret is sealed with XChaCha20-Poly1305 under the `totp` purpose
key of the `VELD_KEYSTORE_PATH` keystore, or of the keyring when that is
unset, with the user id bound in as associated data. With neither,
enrollment is refused with `TOTP_ENCRYPTION_REQUIRED`.

## Recovery codes

Ten codes are issued at 2FA enrollment, formatted as
`ABCDE-FGHJ-KMNPQ-XY` (16 Crockford-style base32 characters → 80 bits of
entropy, presented in 5-4-5-2 groups). The plaintext is returned exactly
once; only Argon2id hashes are stored. Redeem semantics are one-shot:
each successful redemption removes the matching hash from the user's
list.

A successful `/api/user_auth/recover` call:

1. consumes one recovery code,
2. resets the password to `new_password`,
3. clears the TOTP secret (user must re-enroll),
4. wipes every existing session for the user.

Re-enrolling 2FA mints a fresh batch of 10 codes — the old batch (and
any unused codes from it) is replaced wholesale.

## Admin bootstrap — why no self-service promotion

The very first registration call (empty user table) silently issues an
`Admin` role to bootstrap the system. Every subsequent registration
requires the caller to present a valid session belonging to an existing
Admin (`Authorization: Bearer <token>`). A non-Admin session yields
`403 FORBIDDEN`.

There is **no self-service path to promote a non-Admin to Admin**, and
the role field is not editable through the API today. This is
deliberate: in a self-hosted edge deployment, the operator is the one
person with shell access to the machine, and granting Admin via an
authenticated API call would mean a compromised regular session could
escalate privilege without any out-of-band check. Today, the only way
to mint another Admin is for an existing Admin to call `/register` and
register a new admin account directly (the role assignment happens in
the handler; a future iteration will accept `role` as a request field
for admin-initiated registration). The current implementation
auto-assigns `User` to every admin-bootstrapped registration; an admin
who needs another admin must currently do this by hand (RocksDB CLI),
which is the right speed bump for a security-sensitive transition.

If a deployment loses access to its last Admin account, the operator
must intervene at the storage layer (drop the user_auth CF or rewrite
the role field of an existing record). This is the same posture as
losing root on a Linux box: rebuild from media. The blast radius of a
runaway promotion bug — silently turning every user into an Admin —
is far larger than the cost of one manual recovery, so we lean towards
the harder-to-misuse design.

## Module layout

- `src/user_auth/mod.rs` — public types (`UserRecord`, `UserRole`,
  `AuthError`, `SessionUser`, `SessionTokenExt`), feature flag check.
- `src/user_auth/password.rs` — Argon2id hash / verify.
- `src/user_auth/totp.rs` — RFC 6238 TOTP wrapper.
- `src/user_auth/recovery_codes.rs` — generation / redeem.
- `src/user_auth/session.rs` — token issuance / refresh / hashing.
- `src/user_auth/store.rs` — RocksDB persistence under the `user_auth` CF.
- `src/user_auth/runtime.rs` — long-lived runtime (store, login limiter,
  step-up elevations, TOTP encryptor) wired into `MultiUserMemoryManager`.
- `src/handlers/user_auth.rs` — Axum handlers + session-bearer
  middleware.
- `src/handlers/user_admin.rs` — administering logins, self-service, the
  audit trail, and the step-up check.
- `src/handlers/keys.rs` — key management over HTTP.
