# Security

Veld ships **secure by default**. Operators who need looser behaviour must opt
in explicitly via environment variables. This page summarises the security
posture; the canonical document is [SECURITY.md](https://github.com/Portll/veld/blob/main/SECURITY.md)
in the repo.

## Reporting a vulnerability

- **Email:** [john@portll.net](mailto:john@portll.net)
- **GitHub:** [Security Advisories](https://github.com/Portll/veld/security/advisories/new) (private)

**Do not** open public GitHub issues for security vulnerabilities. Initial
response within 48 hours; status update within 7 days; fix timeline depends
on severity.

## Secure-by-default environment variables

All behaviour-changing security controls have a safe default. Override only
in trusted environments.

| Variable | Default | Loosened behaviour when changed |
|---|---|---|
| `VELD_ALLOW_UNSIGNED_WEBHOOKS` | `false` | `true` processes webhooks even when no `*_WEBHOOK_SECRET` is configured. Default rejects with 503. |
| `VELD_PUBLIC_RATE_LIMIT` | `true` | `false` exempts non-probe public routes from rate limiting. Probe routes are never rate-limited regardless. |
| `VELD_METRICS_PUBLIC` | `false` | `true` exposes `/metrics` without authentication. Default requires `Authorization: Bearer <key>`. |
| `VELD_ENFORCE_HTTPS` | `false` | `true` rejects insecure `http://` overrides for `LINEAR_API_URL`/`GITHUB_API_URL`. Default warns only. |
| `VELD_ADMIN_API_KEY` | unset | Set to a secret string to enable `/api/admin/*` endpoints. Separate from `VELD_API_KEYS`. |

## Webhook security

Linear and GitHub webhook handlers perform HMAC-SHA256 signature verification.
Missing or invalid signatures return 401. When no `*_WEBHOOK_SECRET` is
configured, webhooks are rejected by default (503) to prevent unauthenticated
memory poisoning. Override with `VELD_ALLOW_UNSIGNED_WEBHOOKS=true` only in
trusted environments.

## API-key authentication

All routes except `/health/*` probes require an API key. Two mechanisms:

- **`VELD_API_KEYS`** — comma-separated list of accepted keys. Each request
  must carry `Authorization: Bearer <key>` or `X-API-Key: <key>`.
- **Phase C user auth** — password + TOTP + recovery codes at
  `/api/user_auth/*` for end-user accounts. Session tokens issued by this
  flow are separate from the API-key mechanism. Both can coexist.

A leaked user API key does **not** grant access to admin endpoints. Admin
access requires the separate `VELD_ADMIN_API_KEY`.

## Public-route safety contract

The router enforces a structural invariant: no handler mounted on a public
route (no auth required) may read `?user_id=` for per-tenant data. The
`public_router_has_no_per_user_handlers` test fails if a new public handler
violates this. See [src/handlers/router.rs](https://github.com/Portll/veld/blob/main/src/handlers/router.rs)
for `PUBLIC_PATHS` / `PROBE_PATHS` constants.

## Data at rest

By default, veld data is stored unencrypted in the platform's data directory
(see [Configuration reference](reference/config.md)). For at-rest encryption,
use filesystem-level encryption (BitLocker, FileVault, LUKS) or a service
like Cloudflare One that encrypts the underlying volume. Veld also has an
`encryption` module ([src/encryption.rs](https://github.com/Portll/veld/blob/main/src/encryption.rs))
for sensitive-field encryption inside memory records — see source for current
status.

## Distribution (fortress feature)

The `fortress` feature flag produces a distribution build with:

- Tracing eliminated (no log output)
- Error messages replaced by codes
- Custom panic handler (no stack traces leaked)
- Anti-debug checks
- Integrity checks

```sh
cargo build --release --features fortress
```

Use this for distribution to untrusted environments. Not recommended for
development builds — debugging is much harder.

## See also

- [Multi-tenant](guides/multi-tenant.md) — per-tenant isolation model
- [Configuration reference](reference/config.md) — full env var list
- [SECURITY.md](https://github.com/Portll/veld/blob/main/SECURITY.md) — canonical policy
