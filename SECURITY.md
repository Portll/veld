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
