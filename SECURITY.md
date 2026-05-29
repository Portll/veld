# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.7.6-unstable | Best-effort internal stabilization branch |
| < 0.7.6 | :x: |

## Reporting a Vulnerability

If you discover a security vulnerability in Veld - Agentic Memory, please report it privately. This branch is still an internal stabilization line and should not be treated as a public security-hardened release.

1. **Email**: Send details to john@portll.net
2. **GitHub**: Use [Security Advisories](https://github.com/Portll/veld/security/advisories/new) to report privately

**What to include:**
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes

**Response timeline:**
- Initial response within 48 hours
- Status update within 7 days
- Fix timeline depends on severity

**Do not:**
- Open public issues for security vulnerabilities
- Disclose publicly before a fix is available

We appreciate responsible disclosure.

## Secure-by-Default Posture

All behavior-changing security controls ship **secure by default**. Operators who need to restore previous behavior must opt in explicitly with an environment variable.

| Environment Variable | Default | Effect when changed |
|---|---|---|
| `VELD_ALLOW_UNSIGNED_WEBHOOKS` | `false` | Set to `true` to process webhooks when no `*_WEBHOOK_SECRET` is configured (not recommended). Default rejects with 503. |
| `VELD_PUBLIC_RATE_LIMIT` | `true` | Set to `false` to exempt non-probe public routes from rate limiting. Probe routes (`/health*`) are never rate-limited regardless. |
| `VELD_METRICS_PUBLIC` | `false` | Set to `true` to expose `/metrics` without authentication (for unauthenticated Prometheus scrapers). Default requires `Authorization: Bearer <key>`. |
| `VELD_ENFORCE_HTTPS` | `false` | Set to `true` to reject insecure `http://` overrides for `LINEAR_API_URL`/`GITHUB_API_URL` (falls back to the compiled-in HTTPS default). Default warns only. |
| `VELD_ADMIN_API_KEY` | unset | Set to a secret string to enable `/api/admin/*` endpoints. These endpoints use a separate key from `VELD_API_KEYS` — a leaked user key does not grant admin access. |

### Webhook Security

Webhook handlers perform HMAC-SHA256 signature verification. A missing or invalid signature is rejected with 401. When no `*_WEBHOOK_SECRET` is configured, webhooks are rejected by default (503) to prevent unauthenticated memory poisoning. Override with `VELD_ALLOW_UNSIGNED_WEBHOOKS=true` only in trusted environments.

### Rate Limiting

Public routes are rate-limited by default (per-IP token bucket). Kubernetes liveness/readiness probes (`/health*`) are served from a separate router that is never rate-limited, so a saturated public rate limit cannot block pod health checks.

### Fact-Purge Surface (`/api/facts/preview-purge` and `/api/facts/purge`)

The destructive purge subsystem (see `docs/src/decisions/0005-facts-purge-and-narratives.md`) has three properties that operators should understand:

- **Pattern hashing.** Substring patterns passed to either route are hashed with SHA-256 before being stored on the fact record (`PurgeReason::PatternMatch { pattern_hash }`) and in the audit log. The raw pattern never persists. NOTE: hashing protects against passive log readers but **not** against an attacker who already knows the pattern — they can compute the same hash and confirm a match. Forensic detail beyond the hash lives in the audit-log payload of the purge event.

- **Hosaka collective is unaffected.** The multi-tenant `collective_store` aggregates retrieval-weight feedback events, not individual facts. Purging a fact in user A does **not** propagate to user B's collective view, and there is no fact-level cross-user state to clean up. Aggregated weights derived from past feedback events on purged facts remain in the population prior — this is intentional, not a leak.

- **Backup-restore re-introduces purged facts.** A backup taken before a purge will, on restore, re-introduce the purged records (their `purged_at` field is preserved, but the reaper window resets from the restore time). Operators who want purges to survive restore must either replay the audit log against the restored snapshot, or take a fresh backup after the purge.

### Time-Travel Queries

`SemanticFactStore::as_of(at, ...)` returns facts that were valid at the given instant. Purged facts (`purged_at IS NOT NULL`) are **always** excluded regardless of whether `at` predates the purge timestamp — administrative removal is not reversible via point-in-time query. This is the time-travel-leak guard from the breakers evaluation (`evaluations/breakers-revised-plan-p1-2026-05-29.json`).
