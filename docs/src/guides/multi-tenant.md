# Multi-tenant

The `multi-tenant` feature flag enables shared-veld operation across multiple
isolated users.

```sh
cargo build --release --features multi-tenant
```

## What this gives you

| Capability | Description |
|---|---|
| Hosaka collective store | Shared knowledge store accessible across tenants with explicit access control |
| Per-tenant maintenance | Decay, consolidation, and metrics scoped per tenant |
| PII policy | Per-tenant rules for handling personally identifiable information |
| Per-tenant rate limiting | Independent limits per API key (separate from global) |

Source: [`src/extensions/`](https://github.com/Portll/veld/tree/main/src/extensions).

## Tenant model

- Every request carries an API key.
- The auth middleware ([src/auth.rs](https://github.com/Portll/veld/blob/main/src/auth.rs))
  resolves the API key to a user / tenant pair.
- All memory operations are scoped to the tenant; cross-tenant access requires
  explicit ACL configuration in the collective store.

## Phase C user auth

User-facing authentication (password + TOTP + recovery codes) is also
multi-tenant-aware. Endpoints at `/api/user_auth/*`:

- `POST /api/user_auth/register`
- `POST /api/user_auth/login`
- `POST /api/user_auth/recover`
- `POST /api/user_auth/2fa/enroll`
- `POST /api/user_auth/2fa/confirm`
- `POST /api/user_auth/logout`

Session tokens issued by user-auth are separate from the API-key
mechanism — both can coexist.

## When NOT to use multi-tenant

For single-user setups (developer desktop, robot edge node), the
multi-tenant feature is unnecessary overhead. Keep the default build.

## See also

- [Deploying](deploying.md)
- [Configuration reference](../reference/config.md)
