//! IPv4 loopback callback server.
//!
//! Spawned once per CLI `veld oauth login` invocation. Binds
//! `127.0.0.1:0` (kernel picks an ephemeral port), accepts a single
//! `GET /?code=...&state=...` request, sends the parsed values through
//! a `oneshot` channel, and returns a short success page to the
//! browser so the user knows it worked.
//!
//! # Google redirect-URI exact-match
//!
//! Google requires the redirect URI to exactly match the URI
//! registered in the Cloud Console. For installed/desktop apps the
//! published guidance is to register `http://127.0.0.1` (no port);
//! at runtime any loopback port is accepted as long as the host is
//! `127.0.0.1`. [`run_loopback_once`] therefore returns the full
//! `http://127.0.0.1:<port>` URL it bound to so the caller can pass
//! it back to `oauth2`'s `authorize_url` / `exchange_code` plumbing.
//!
//! # IPv4 only
//!
//! Windows Firewall has historically treated `[::1]` and `127.0.0.1`
//! as distinct sources; Google's Cloud Console UI also lists the
//! loopback in IPv4 form. Binding v4 exclusively avoids both pitfalls.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use super::OauthError;

/// Bytes returned to the browser on a successful callback. Plain HTML
/// so it works without JavaScript; tells the user to return to the
/// terminal.
const SUCCESS_HTML: &str = concat!(
    "HTTP/1.1 200 OK\r\n",
    "Content-Type: text/html; charset=utf-8\r\n",
    "Connection: close\r\n\r\n",
    "<!doctype html><html><head><title>veld oauth — done</title></head>",
    "<body style='font-family:system-ui;text-align:center;padding:4em'>",
    "<h2>Authentication received</h2>",
    "<p>You can close this tab and return to the terminal.</p>",
    "</body></html>",
);

const ERROR_HTML: &str = concat!(
    "HTTP/1.1 400 Bad Request\r\n",
    "Content-Type: text/html; charset=utf-8\r\n",
    "Connection: close\r\n\r\n",
    "<!doctype html><html><body>",
    "<h2>OAuth callback malformed</h2>",
    "<p>Missing <code>code</code> or <code>state</code> — try logging in again.</p>",
    "</body></html>",
);

/// What [`run_loopback_once`] forwards to the calling task once the
/// browser GET arrives.
#[derive(Debug)]
pub struct CallbackResult {
    pub code: String,
    pub state: String,
}

/// Bind a single-use loopback listener and spawn a tokio task that
/// reads exactly one HTTP request from it. Returns:
///
/// - the bound `SocketAddr` (port-only — host is always `127.0.0.1`),
/// - the full `http://127.0.0.1:<port>` redirect URL the caller passes
///   to `oauth2`'s `authorize_url` builder,
/// - a [`oneshot::Receiver`] that resolves with the parsed callback or
///   an [`OauthError`] if the request was malformed or the timeout fired.
pub async fn run_loopback_once(
    timeout: Duration,
) -> Result<(SocketAddr, String, oneshot::Receiver<Result<CallbackResult, OauthError>>), OauthError>
{
    // Try the bind up to three times — the kernel can hand back a port
    // a firewall has already reserved on certain Windows configs.
    let mut last_err: Option<std::io::Error> = None;
    let mut listener: Option<TcpListener> = None;
    for _ in 0..3 {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        match TcpListener::bind(addr).await {
            Ok(l) => {
                listener = Some(l);
                break;
            }
            Err(e) => last_err = Some(e),
        }
    }
    let listener = listener.ok_or_else(|| {
        last_err
            .map(|e| OauthError::Io(e.to_string()))
            .unwrap_or(OauthError::LoopbackBindFailed)
    })?;
    let bound = listener.local_addr()?;
    let redirect_uri = format!("http://127.0.0.1:{}", bound.port());

    let (tx, rx) = oneshot::channel();
    tokio::spawn(async move {
        let result = tokio::time::timeout(timeout, accept_once(listener)).await;
        let payload = match result {
            Ok(inner) => inner,
            Err(_) => Err(OauthError::CallbackTimeout),
        };
        let _ = tx.send(payload);
    });

    Ok((bound, redirect_uri, rx))
}

async fn accept_once(listener: TcpListener) -> Result<CallbackResult, OauthError> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let (mut socket, _peer) = listener.accept().await?;
    let mut buf = [0u8; 4096];
    let n = socket.read(&mut buf).await?;
    let req = std::str::from_utf8(&buf[..n]).unwrap_or("");

    // Parse just the first request line: "GET /?code=...&state=... HTTP/1.1"
    let first_line = req.lines().next().unwrap_or("");
    let target = first_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| OauthError::Oauth2("loopback request missing target".into()))?;

    let parsed = match parse_callback_query(target) {
        Some(p) => p,
        None => {
            let _ = socket.write_all(ERROR_HTML.as_bytes()).await;
            return Err(OauthError::Oauth2(
                "loopback callback missing `code` or `state`".into(),
            ));
        }
    };

    let _ = socket.write_all(SUCCESS_HTML.as_bytes()).await;
    Ok(parsed)
}

/// Extract `code` and `state` from a request target like
/// `/?code=abc&state=xyz`. Returns `None` if either is missing.
pub(crate) fn parse_callback_query(target: &str) -> Option<CallbackResult> {
    let q_idx = target.find('?')?;
    let query = &target[q_idx + 1..];
    let mut code: Option<String> = None;
    let mut state: Option<String> = None;
    for pair in query.split('&') {
        let mut it = pair.splitn(2, '=');
        let k = it.next()?;
        let v = it.next()?;
        let v_decoded = percent_decode(v);
        match k {
            "code" => code = Some(v_decoded),
            "state" => state = Some(v_decoded),
            _ => {}
        }
    }
    Some(CallbackResult {
        code: code?,
        state: state?,
    })
}

/// Minimal percent-decoder — handles `%NN` and `+` (form-encoded
/// spaces) without pulling in a URL crate. Bad escapes pass through
/// verbatim.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = hex_nibble(bytes[i + 1]);
                let lo = hex_nibble(bytes[i + 2]);
                match (hi, lo) {
                    (Some(h), Some(l)) => {
                        out.push((h << 4) | l);
                        i += 3;
                    }
                    _ => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_callback_extracts_fields() {
        let r = parse_callback_query("/?code=abc&state=xyz").unwrap();
        assert_eq!(r.code, "abc");
        assert_eq!(r.state, "xyz");
    }

    #[test]
    fn parse_callback_missing_field_is_none() {
        assert!(parse_callback_query("/?code=abc").is_none());
        assert!(parse_callback_query("/?state=xyz").is_none());
        assert!(parse_callback_query("/").is_none());
    }

    #[test]
    fn parse_callback_percent_decodes() {
        let r = parse_callback_query("/?code=ab%2Bc&state=%2Fxyz").unwrap();
        assert_eq!(r.code, "ab+c");
        assert_eq!(r.state, "/xyz");
    }
}
