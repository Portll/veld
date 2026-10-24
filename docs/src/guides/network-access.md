# Reaching Veld from another machine

`veld server` listens on `127.0.0.1:3030` in plain HTTP by default. Every protected request
carries the API key (`X-API-Key` or `Authorization: Bearer`), the replication routes also
carry the admin key, and the bodies carry memory content. This page covers the three ways
to reach the server from another machine, and how each client is configured to trust it.

| Option | Veld listens on | Who can reach it | Certificates |
|---|---|---|---|
| [Tailscale Serve](#option-1-tailscale-serve) | loopback only | devices on your tailnet | issued by Tailscale |
| [Native TLS listener](#option-2-native-tls-listener) | loopback HTTP plus a TLS port | whoever can reach the TLS port | yours, optionally with client certificates |
| [Reverse proxy](#option-3-tls-terminating-reverse-proxy) | loopback, or an interface the proxy reaches | whoever can reach the proxy | managed by the proxy |

## Why not `VELD_HOST=0.0.0.0` alone

`VELD_HOST=0.0.0.0` makes the HTTP listener answer on every interface, still in plain HTTP:

- The API key, the admin key and memory content cross the network in cleartext. Anyone on
  the path can read them and reuse the key.
- Rate limiting switches on. Outside production mode, a server bound to `127.0.0.1`,
  `localhost` or `::1` runs with no rate limit unless `VELD_RATE_LIMIT` is set; any other
  bind gets the default of 4000 requests per second, counted per client IP address
  (`VELD_RATE_LIMIT`, `VELD_RATE_BURST`).
- The key-management routes (`/api/user_auth/admin/keys/*`, which carry passphrases) answer
  `403 TLS_REQUIRED` on a non-loopback bind unless `VELD_TLS_ACK=true`.

Use one of the options below instead.

## Option 1: Tailscale Serve

Keep Veld on loopback and let Tailscale terminate TLS. On the machine running Veld:

```sh
tailscale serve --bg --https=3030 http://127.0.0.1:3030
```

Clients on the tailnet use `https://<machine>.<tailnet>.ts.net:3030`. The certificate is
issued by Tailscale for that name from a public CA, so clients need no extra trust
configuration; HTTPS certificates must be enabled for the tailnet. `tailscale serve status`
shows the mapping and `tailscale serve reset` removes it.

Do not use `tailscale funnel`: it publishes the port to the internet.

Veld sees every request arrive from `127.0.0.1`. Its bind is loopback, so rate limiting
stays off and the key-management routes accept requests; the API key still gates every
protected route, and the tailnet's access rules decide which devices reach the port.

Pick this when every client is on the same tailnet. Nothing changes on the Veld side.

## Option 2: Native TLS listener

Veld can run a TLS listener beside the loopback HTTP listener, which stays as it is:

| Variable | Meaning |
|---|---|
| `VELD_TLS_CERT` | Server certificate chain, PEM file |
| `VELD_TLS_KEY` | Server private key, PEM file |
| `VELD_TLS_PORT` | Port of the TLS listener |
| `VELD_TLS_HOST` | Address the TLS listener binds (default `0.0.0.0`) |
| `VELD_TLS_CLIENT_CA` | Optional. PEM bundle of the CA that signs client certificates; setting it enables mutual TLS on the TLS listener |

The server's entry in the [instance registry](../architecture/instance-registry.md) gains
`tls_url` for the TLS listener. `base_url` stays the loopback HTTP URL, so the monitor and
other local tools keep connecting over loopback.

The certificate must name the host clients connect to in its subjectAltName. A leaf issued
for `m5pro-2.local` is reached as `https://m5pro-2.local:<port>`; connecting by IP address
fails verification unless the certificate lists that address.

With mutual TLS, clients still send the API key; a client certificate does not replace
it. The MCP server and the hooks do not present client certificates, so they cannot use a
TLS listener that requires one. The replication follower can (see below).

Pick this when clients are outside a tailnet, or on a LAN, and you can issue a certificate,
for example from a private CA. It is the only option here that can require client
certificates.

## Option 3: TLS-terminating reverse proxy

Caddy, nginx, a cloud load balancer or Cloudflare Tunnel terminate TLS and forward plain
HTTP to Veld.

- **Proxy on the same machine:** forward to `http://127.0.0.1:3030` and leave Veld on
  loopback.
- **Proxy on another machine or in another container:** Veld has to bind an interface the
  proxy can reach (`VELD_HOST`), and the hop between the proxy and Veld is plain HTTP; keep
  it on a private network. Set `VELD_TLS_ACK=true` to state that TLS terminates in front of
  Veld. It silences the production-mode warning about a non-loopback bind and lets the
  key-management routes run on that bind.
- With a non-loopback bind, rate limiting counts requests by the peer address, which is the
  proxy's: every client behind it shares one budget.

Pick this when a proxy already fronts your services, or you need certificates from a
public CA (ACME) on a public name.

## Client configuration

### Rust clients: `VELD_CA_CERT`

`VELD_CA_CERT` is the path to a PEM file holding one or more CA certificates. They are
added to the roots compiled into the binary (the webpki list); they do not replace them,
and the operating system's certificate store is not read. It is read by:

- the replication follower, when a follow starts (`VELD_REPLICATE_FROM` at boot, or
  `POST /api/admin/replication/follow`);
- status-core, on every request: the Veld GUI (monitor), the status TUI and the status
  widget;
- the terminal dashboard (`veld-tui`), once at start. A bad bundle stops it before the
  screen opens. Its `statusline` mode, whose only output is the line Claude Code shows,
  falls back to the built-in roots instead, so its fire-and-forget report fails like any
  unreachable server.

A file that cannot be read, holds no certificate or does not parse is an error naming the
variable and the path; it is never skipped. The follower then does not follow: at boot the
log shows `Replication: not following at boot: VELD_CA_CERT=…`, and the follow route
answers `500 replication_failed` with the message in `detail`. status-core returns the
error from each call instead of connecting without the CA.

Use absolute paths. The path is opened as given, and `~` is not expanded.

### Replication follower with a client certificate

| Variable | Meaning |
|---|---|
| `VELD_REPLICATE_CLIENT_CERT` | The follower's certificate chain, PEM, leaf first |
| `VELD_REPLICATE_CLIENT_KEY` | Its private key, PEM: PKCS#8, PKCS#1 (RSA) or SEC1 (EC) |

Set both or neither. One without the other is an error naming both variables, reported
where the follow is reported, as above; so is a file that cannot be read or does not hold
a certificate or a key. The files are read when a follow starts, so after replacing them,
follow again or restart the server.

A leader on another host must still be `https`. `VELD_REPLICATE_INSECURE_HTTP=1` accepts
plain HTTP on a private link, and then no certificate is presented.

The follow route refuses a leader that already follows this instance by comparing URLs: it
catches a leader that follows this instance's loopback URL, not one that follows its TLS
URL or a proxy URL. Do not point two instances at each other.

### MCP server and hooks

| Variable | Meaning |
|---|---|
| `VELD_API_URL` | Server URL, e.g. `https://m5pro-2.local:3443` (default `http://127.0.0.1:3030`) |
| `VELD_API_KEY` | API key. The MCP server exits without one for a non-local URL; the hooks fall back to `api_key` in Veld's `config.toml` |
| `NODE_EXTRA_CA_CERTS` | PEM file of extra CA certificates, for a server certificate from a private CA |

The MCP server warns on a non-local `http://` URL unless `VELD_ALLOW_HTTP=true`. Its
streaming connection uses the same URL with the scheme changed, so `https://` becomes
`wss://`.

The hooks run under Bun (`bun run …/memory-hook.ts` in `hooks/claude-settings.json`). The
repository's `.mcp.json` also starts the MCP server with Bun; the published `veld-mcp`
entry point is built for Node. Both runtimes read `NODE_EXTRA_CA_CERTS` once, when the
process starts (checked with Bun 1.3.14 and Node 26: `tls.getCACertificates("extra")`
returns the file's certificates), so it has to be in the environment the process is
started with: the `env` block of the MCP server's configuration, and the environment of
the Claude Code process, which the hooks inherit. Bun also accepts `--use-system-ca` to
trust the operating system's store.

Do not set `NODE_TLS_REJECT_UNAUTHORIZED=0`; it turns verification off for every
connection the process makes.

## Example: a private CA in `~/.substrate/tls`

The CA certificate is `~/.substrate/tls/ca.pem`, and it signed `server.pem` (the leaf for
`m5pro-2.local`, followed by any intermediates) with its key `server.key`. The port 3443
and the home directory `/Users/you` are example values.

On `m5pro-2`, in the server's environment:

```sh
VELD_TLS_CERT=/Users/you/.substrate/tls/server.pem
VELD_TLS_KEY=/Users/you/.substrate/tls/server.key
VELD_TLS_PORT=3443
```

On another machine, check the chain before configuring clients. The last line should read
`Verify return code: 0 (ok)`:

```sh
openssl s_client -connect m5pro-2.local:3443 -servername m5pro-2.local \
  -CAfile /Users/you/.substrate/tls/ca.pem </dev/null
```

MCP server and hooks:

```sh
VELD_API_URL=https://m5pro-2.local:3443
VELD_API_KEY=<api key>
NODE_EXTRA_CA_CERTS=/Users/you/.substrate/tls/ca.pem
```

A follower:

```sh
VELD_REPLICATE_FROM=https://m5pro-2.local:3443
VELD_REPLICATE_ADMIN_KEY=<leader's admin key>
VELD_REPLICATE_API_KEY=<leader's api key>
VELD_CA_CERT=/Users/you/.substrate/tls/ca.pem
```

To require client certificates, add `VELD_TLS_CLIENT_CA=/Users/you/.substrate/tls/ca.pem`
on the server, issue the follower a leaf from the same CA, and add to the follower:

```sh
VELD_REPLICATE_CLIENT_CERT=/Users/you/.substrate/tls/replica.pem
VELD_REPLICATE_CLIENT_KEY=/Users/you/.substrate/tls/replica.key
```

`openssl s_client` with `-cert` and `-key` added checks that pair the same way. The MCP
server and hooks cannot connect to the TLS listener once it requires client certificates;
point them at a Tailscale Serve URL or a proxy instead.

## See also

- [Deploying](deploying.md)
- [Live replication](../architecture/replication.md)
- [Security](../security.md)
