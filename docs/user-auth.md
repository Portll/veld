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
| `VELD_USER_AUTH_ENABLED`  | `true` / `1` / `yes` / `on` enables the surface. Default: off (no CF, routes 404).           |
| `VELD_ENCRYPTION_KEY`     | Reused from the memory subsystem. When set, TOTP secrets at rest are AES-256-GCM encrypted. |
| `VELD_ENV`                | Reused. `production` makes a missing `VELD_ENCRYPTION_KEY` refuse 2FA enrollment.            |

No other env vars are introduced. The login throttle (5 attempts / 15 min,
per-username) is hard-coded — admins can clear it by restarting the
server or, in the future, an admin endpoint analogous to
`/api/admin/reset-rate-limit`.

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
| 400  | `TOTP_ENCRYPTION_REQUIRED`   | Production mode + no `VELD_ENCRYPTION_KEY` blocks enroll.  |
| 403  | `FORBIDDEN`                  | Non-admin tried to register a new user.                    |
| 409  | `USERNAME_TAKEN`             | Registration: username already maps to a user.             |
| 429  | `TOO_MANY_ATTEMPTS`          | Per-username throttle tripped (5 / 15 min).                |
| 404  | `USER_AUTH_DISABLED`         | Feature flag is off. (Routes return 404 directly.)         |

## Sessions

Tokens are 32 random bytes encoded as URL-safe base64 (no padding), 43
chars on the wire. The server only persists the SHA-256 digest, so a
database compromise alone cannot resurrect a live session.

Lifetime: 24 hours, **refreshed on every use**. A session that's exercised
every few minutes effectively never expires; one that's idle for 24h
dies. There is no hard cap today — to invalidate everything for a user,
trigger `/api/user_auth/recover` (which wipes all sessions) or restart
the server.

## TOTP

`totp-rs 5.x` with SHA-1 / 30-second / 6-digit profile — the baseline
every authenticator app accepts. Skew tolerance is ±1 step (30 s
either side of "now") to handle mildly mis-synced clocks. The
provisioning URI uses issuer `"Veld"` and the username as the account
label (URL-encoded by `totp-rs`).

The HMAC secret is encrypted with `FieldEncryptor` (AES-256-GCM)
whenever `VELD_ENCRYPTION_KEY` is configured. In development mode with
no key, secrets are stored as raw bytes and a WARN log fires. In
production mode (`VELD_ENV=production`) without the key, enrollment is
refused with `TOTP_ENCRYPTION_REQUIRED`.

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
- `src/user_auth/runtime.rs` — long-lived runtime (store + login limiter
  + field encryptor) wired into `MultiUserMemoryManager`.
- `src/handlers/user_auth.rs` — Axum handlers + session-bearer
  middleware.
