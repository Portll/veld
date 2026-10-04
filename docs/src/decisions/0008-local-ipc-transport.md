---
id: 0008
title: Whether veld takes upstream's authenticated local IPC transport
status: proposed
date: 2026-10-04
---

# 0008 — Whether veld takes upstream's authenticated local IPC transport

Upstream shodh-memory #399 (merge `2843d91f`, 2026-07-15) adds a second way into the server: an
authenticated local IPC endpoint beside the HTTP listener. W1 of the upstream catch-up
(`docs/upstream/PLAN.md`) opens with this decision; the ledger row stays `pending` until it is made.

## What #399 does

- **One local endpoint, the same route table.** The server binds a Unix-domain socket (Linux,
  macOS) or a named pipe (Windows) at startup and dispatches each request into the same Axum router
  the HTTP listener serves. Defaults: `~/.local/share/shodh/shodh-memory.sock`,
  `~/Library/Application Support/shodh/shodh-memory.sock`, `\\.\pipe\shodh-memory-<user SID>`;
  `SHODH_IPC_ENDPOINT` overrides.
- **On by default, non-fatal.** A bind failure is logged and the server carries on over HTTP.
  `SHODH_IPC_ENABLED=false` turns it off; `SHODH_IPC_REQUIRED=true` makes server and Rust client
  fail closed.
- **OS access control.** The socket is 0600 inside a 0700 parent opened with `O_NOFOLLOW`, both
  ends check the peer UID, and stale-socket cleanup checks owner, type, liveness and inode. The
  Windows pipe carries a protected DACL for the current user and LocalSystem, refuses remote
  clients, and both ends check the peer account.
- **Mutual authentication without sending the key.** Protocol v2 is newline-delimited JSON, one
  request per connection, frames capped at 8 MiB. A client first sends `GET /health` with a random
  challenge; the server answers with its per-process nonce and an HMAC proof per configured API
  key. Each request then carries an HMAC over the nonce, request id, method, path and raw body,
  checked before any route or body processing, and each response carries a proof bound to the
  request. The API key itself never crosses the socket.
- **Limits.** At most 16 connections in the pre-authentication stage, two seconds each, and a
  separate dispatch concurrency limit. Streaming routes (SSE, WebSocket) are refused with 501;
  public, webhook and HTML routes are not exposed.
- **Clients.** The Rust MCP client (`shodh serve`) prefers IPC and falls back to HTTP. The
  TypeScript MCP server uses IPC when `SHODH_IPC_ENDPOINT` is set, through a 566-line client, and on
  Windows routes pipe I/O through helper processes of the Rust binary.
- **Size.** 3,924 added lines across 21 files: `src/local_ipc.rs` (2,287), `mcp-server/ipc-client.ts`
  (566) and its tests (352), plus changes to `server.rs`, `cli.rs`, `auth.rs` and the MCP server.

Upstream's own design note lists what it does not do: the streaming and public-route exclusion
lists are kept by hand, because Axum cannot enumerate routes; HTTP middleware (rate limiting, CORS,
security headers) does not run on IPC requests, and new middleware does not cover IPC
automatically; responses lose their headers and content type and are capped at 8 MiB where HTTP has
no cap; and an API key keeps full authority over every `user_id`.

## veld's loopback clients today

Every local client reaches the server over HTTP on `127.0.0.1:3030` with the API key in
`X-API-Key`:

- the menu-bar supervisor (`veld-desktop`), which writes `VELD_API_URL`, `VELD_API_KEY` and
  `VELD_USER_ID` into `~/.config/veld/<user>/connection.env` for each enabled project;
- the TUI (`tui/`) and its event stream;
- the Rust MCP server (`veld serve`, `src/mcp/client.rs`) and the TypeScript MCP server
  (`@veld/memory-mcp`);
- the Claude Code hooks (`hooks/memory-hook.ts`, the shell hooks, `statusline.ts`) and
  `veld hook ...`;
- discovery through `src/instance_registry.rs`, which advertises host and port.

The listener binds loopback only, every protected route sits behind `auth_middleware`, and the
route inventory test (`public_router_paths_match_const`) pins which paths are public.

## What adopting it would mean for veld

- **Auth runs in two places.** veld's `auth_middleware` does more than check a key: under
  `multi-tenant` it binds the key to a tenant and attaches `AuthenticatedUser`, which
  `resolve_request_user_id` enforces on every handler; `/api/user_auth/*` uses session tokens.
  #399 dispatches past the HTTP middleware, so the port has to rebuild the tenant binding inside the
  IPC layer, or an IPC request would reach handlers without the binding HTTP requests carry.
  Upstream's statement that a key has authority over every namespace is not true of veld's
  multi-tenant mode.
- **The middleware stack splits.** Rate limiting (`rate_limit_governance`), security headers and
  request timeouts would apply to HTTP only, and each new layer would need an IPC decision.
- **The route inventory has to cover a second transport.** The hand-kept streaming and public
  lists would join the route and tool auth inventory (PLAN 1.6), so a new route that is public or
  streaming fails a test until it is classified for IPC.
- **Every client gains a second path.** The TypeScript MCP server takes the IPC client; the Rust MCP
  client and `veld hook` take the Rust one; the desktop supervisor would write an endpoint as well
  as a URL into `connection.env`; the shell hooks use `curl` and stay on HTTP. The instance
  registry would advertise the endpoint.
- **The endpoint lands in the data directory.** The default socket sits in
  `~/Library/Application Support/veld/`, which holds the live store, and #399 tightens its parent
  to 0700, which the backup, migrate and boot-guard paths would then see.
- **What it buys.** A second local OS user, or a sandboxed process without the key, cannot use the
  socket, and the key stays off the wire. On loopback HTTP a local process still needs the key, and
  reading loopback traffic already needs root. The gain is real on shared multi-user hosts and
  small on a single-user workstation, which is how veld runs today.

## Options

1. **Do not take it now.** Keep HTTP with the API key as the only local transport. Record the row as
   `declined` with this ADR as the reason, and revisit when a deployment needs OS-level peer
   authentication (a shared host, or tenants that must not see each other's sockets).
2. **Take it as upstream ships it.** On by default, all clients prefer IPC. Largest change and
   largest surface: a second entry point beside HTTP, and the auth and middleware split above.
3. **Take it opt-in, Unix first.** `VELD_IPC_ENABLED` off by default; the IPC dispatch runs veld's
   key and tenant-binding check (one function shared with `auth_middleware`, not a copy); the route
   inventory test classifies every route for IPC; clients use IPC only when an endpoint is
   configured; the Windows named pipe follows once a Windows deployment exists.
4. **Take only the transport-independent parts.** Nothing in #399 improves the HTTP path on its own:
   the HMAC request proof only makes sense with the per-process nonce exchange, so this collapses
   into option 1.

## Recommendation

Option 1 now, with option 3 as the shape if a deployment needs it. Taking #399 means a second
authenticated entry point whose auth and middleware must be kept equal to HTTP's by hand, for a
gain that veld's single-user, loopback-only deployments do not need.

## Consequences

- Until this is decided, the ledger row for #399 stays `pending` with the reason
  "awaiting ADR 0008".
- If option 1 is chosen, the row becomes `declined` naming this ADR, and the quarterly review of
  declined rows (PLAN, phase 4) reconsiders it.
- If option 3 is chosen, the port is its own wave item: the shared auth function and the route
  classification test land first, then the server listener, then each client.
