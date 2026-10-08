//! Passkeys as a keyring factor, through the WebAuthn PRF extension.
//!
//! A passkey with PRF returns the same 32-byte secret for a given credential and salt
//! every time, and only after user verification: Touch ID, Face ID, a security key's
//! PIN. The keyring's master key is wrapped under that secret as a `passkey` factor
//! whose id names the relying party and the credential. No assertion is verified:
//! the secret opening the wrap is the proof, and only the authenticator produces it.
//!
//! The ceremony runs in the browser on a page served on loopback with the relying
//! party `localhost`, so a passkey enrolled here also recovers a keyring when no veld
//! server can start.

use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use axum::extract::{Path as UrlPath, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use tokio::sync::{oneshot, Mutex};
use zeroize::Zeroizing;

pub const RP_LOCALHOST: &str = "localhost";
/// The PRF input. Fixed: the credential, not the salt, is what makes a secret unique.
pub const PRF_SALT: &[u8] = b"veld:passkey-prf:v1";

fn b64u(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn b64u_decode(text: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(text.trim_end_matches('='))
        .map_err(|_| anyhow!("malformed base64url"))
}

/// `passkey:<relying party>:<credential id>`: unlocking needs both back.
pub fn factor_id(rp_id: &str, credential_id: &str) -> String {
    format!("passkey:{rp_id}:{credential_id}")
}

/// The relying party and credential id in a passkey factor id.
pub fn parse_factor_id(id: &str) -> Option<(&str, &str)> {
    let rest = id.strip_prefix("passkey:")?;
    let (rp, credential) = rest.split_once(':')?;
    (!rp.is_empty() && !credential.is_empty()).then_some((rp, credential))
}

pub enum Ceremony {
    /// Create a passkey and take its PRF secret.
    Enrol { label: String },
    /// Take the PRF secret of one of these enrolled passkey factor ids.
    Unlock { factors: Vec<String> },
}

/// A PRF secret and the factor id it belongs to.
pub struct PasskeySecret {
    pub id: String,
    pub secret: Zeroizing<Vec<u8>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PageConfig {
    mode: &'static str,
    rp_id: String,
    salt: String,
    user_id: String,
    user_name: String,
    allow: Vec<String>,
    result: String,
}

#[derive(Deserialize)]
struct PageResult {
    credential_id: Option<String>,
    prf: Option<String>,
    error: Option<String>,
}

/// The ceremony page. `result` is the path it posts `{credential_id, prf}` or
/// `{error}` to. Self-contained: no external script, style or font.
pub fn page(ceremony: &Ceremony, rp_id: &str, result: &str, script_nonce: &str) -> String {
    let mut user_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut user_id);
    let config = match ceremony {
        Ceremony::Enrol { label } => PageConfig {
            mode: "enrol",
            rp_id: rp_id.to_string(),
            salt: b64u(PRF_SALT),
            user_id: b64u(&user_id),
            user_name: label.clone(),
            allow: Vec::new(),
            result: result.to_string(),
        },
        Ceremony::Unlock { factors } => PageConfig {
            mode: "unlock",
            rp_id: rp_id.to_string(),
            salt: b64u(PRF_SALT),
            user_id: String::new(),
            user_name: String::new(),
            allow: factors
                .iter()
                .filter_map(|id| parse_factor_id(id))
                .filter(|(rp, _)| *rp == rp_id)
                .map(|(_, credential)| credential.to_string())
                .collect(),
            result: result.to_string(),
        },
    };
    let config = serde_json::to_string(&config)
        .expect("page config serialises")
        .replace('<', "\\u003c");
    PAGE.replace("{{NONCE}}", script_nonce)
        .replace("{{CONFIG}}", &config)
}

const PAGE: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Veld keyring passkey</title>
<style>
  body { font: 15px/1.5 -apple-system, system-ui, sans-serif; background: #15161e; color: #e6e6ef;
         display: grid; place-items: center; min-height: 100vh; margin: 0; }
  main { max-width: 30rem; padding: 2rem; background: #1d1f2b; border: 1px solid #2b2d3c; border-radius: 10px; }
  h1 { font-size: 1.15rem; margin: 0 0 .75rem; }
  button { font: inherit; padding: .55rem 1.1rem; border-radius: 6px; border: 0; background: #5b7cfa; color: #fff; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  #status { margin-top: 1rem; min-height: 1.5em; }
  .error { color: #f07178; } .done { color: #8bd49c; }
</style>
</head>
<body>
<main>
  <h1 id="title"></h1>
  <p id="intro"></p>
  <button id="go" type="button"></button>
  <p id="status"></p>
</main>
<script type="application/json" id="config">{{CONFIG}}</script>
<script nonce="{{NONCE}}">
"use strict";
const cfg = JSON.parse(document.getElementById("config").textContent);
const enc = (buf) => btoa(String.fromCharCode(...new Uint8Array(buf)))
  .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const dec = (s) => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/")), (c) => c.charCodeAt(0));
const $ = (id) => document.getElementById(id);
const say = (text, cls) => { $("status").textContent = text; $("status").className = cls || ""; };
const enrol = cfg.mode === "enrol";
$("title").textContent = enrol ? "Add a passkey to the Veld keyring" : "Unlock the Veld keyring";
$("intro").textContent = enrol
  ? "Create a passkey (Touch ID, Face ID or a security key). It becomes a way to open the keyring, including when the passphrase is lost."
  : "Use the passkey you enrolled to open the keyring.";
$("go").textContent = enrol ? "Create passkey" : "Use passkey";

async function post(body) {
  const res = await fetch(cfg.result, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error("veld refused the result (" + res.status + ")");
}

let pending = null;
async function assertion(allow) {
  const got = await navigator.credentials.get({ publicKey: {
    rpId: cfg.rpId,
    challenge: crypto.getRandomValues(new Uint8Array(32)),
    allowCredentials: allow.map((id) => ({ type: "public-key", id })),
    userVerification: "required",
    extensions: { prf: { eval: { first: dec(cfg.salt) } } },
  }});
  const prf = got.getClientExtensionResults().prf;
  if (!prf || !prf.results || !prf.results.first) {
    throw new Error("this passkey did not return a PRF secret; its authenticator does not support the PRF extension");
  }
  return { credential_id: enc(got.rawId), prf: enc(prf.results.first) };
}

async function run() {
  $("go").disabled = true;
  try {
    let result;
    if (pending) {
      result = await assertion([pending]);
    } else if (enrol) {
      const cred = await navigator.credentials.create({ publicKey: {
        rp: { id: cfg.rpId, name: "Veld" },
        user: { id: dec(cfg.userId), name: cfg.userName, displayName: cfg.userName },
        challenge: crypto.getRandomValues(new Uint8Array(32)),
        pubKeyCredParams: [{ type: "public-key", alg: -7 }, { type: "public-key", alg: -257 }],
        authenticatorSelection: { residentKey: "preferred", userVerification: "required" },
        extensions: { prf: { eval: { first: dec(cfg.salt) } } },
      }});
      const prf = cred.getClientExtensionResults().prf;
      if (prf && prf.enabled === false) {
        throw new Error("this authenticator does not support the PRF extension, so it cannot hold a key; try another passkey provider or security key");
      }
      if (prf && prf.results && prf.results.first) {
        result = { credential_id: enc(cred.rawId), prf: enc(prf.results.first) };
      } else {
        // Some authenticators return the PRF secret only on assertion, which needs a
        // second gesture.
        pending = cred.rawId;
        $("go").textContent = "Confirm passkey";
        $("go").disabled = false;
        say("Passkey created. Confirm it once more to finish.");
        return;
      }
    } else {
      if (cfg.allow.length === 0) throw new Error("no passkey is enrolled for " + cfg.rpId);
      result = await assertion(cfg.allow.map(dec));
    }
    await post(result);
    say("Done. You can close this tab.", "done");
  } catch (e) {
    say(e.message || String(e), "error");
    try { await post({ error: e.message || String(e) }); } catch (_) {}
  }
}
$("go").addEventListener("click", run);
</script>
</body>
</html>
"#;

struct Waiting {
    nonce: String,
    script_nonce: String,
    page: String,
    sender: Mutex<Option<oneshot::Sender<PageResult>>>,
}

async fn serve_page(
    State(state): State<std::sync::Arc<Waiting>>,
    UrlPath(nonce): UrlPath<String>,
) -> Response {
    if nonce != state.nonce {
        return StatusCode::NOT_FOUND.into_response();
    }
    let csp = format!(
        "default-src 'none'; script-src 'nonce-{}'; style-src 'unsafe-inline'; \
         connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        state.script_nonce
    );
    (
        [
            (header::CACHE_CONTROL, "no-store".to_string()),
            (header::CONTENT_SECURITY_POLICY, csp),
        ],
        Html(state.page.clone()),
    )
        .into_response()
}

async fn take_result(
    State(state): State<std::sync::Arc<Waiting>>,
    UrlPath(nonce): UrlPath<String>,
    Json(result): Json<PageResult>,
) -> StatusCode {
    if nonce != state.nonce {
        return StatusCode::NOT_FOUND;
    }
    match state.sender.lock().await.take() {
        Some(sender) => {
            let _ = sender.send(result);
            StatusCode::NO_CONTENT
        }
        None => StatusCode::CONFLICT,
    }
}

fn open_browser(url: &str) {
    let opener = if cfg!(target_os = "macos") {
        Some(("open", vec![url]))
    } else if cfg!(target_os = "windows") {
        Some(("cmd", vec!["/C", "start", "", url]))
    } else {
        Some(("xdg-open", vec![url]))
    };
    if let Some((program, args)) = opener {
        let _ = std::process::Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }
}

/// Turn what the page posted into a secret for one of the expected factor ids.
fn accept(ceremony: &Ceremony, result: PageResult) -> Result<PasskeySecret> {
    if let Some(error) = result.error {
        bail!("the browser reported: {error}");
    }
    let credential = result
        .credential_id
        .ok_or_else(|| anyhow!("the page posted no credential"))?;
    let secret = Zeroizing::new(b64u_decode(
        result
            .prf
            .as_deref()
            .ok_or_else(|| anyhow!("the page posted no PRF secret"))?,
    )?);
    if secret.len() != 32 {
        bail!("a PRF secret is 32 bytes, the page posted {}", secret.len());
    }
    let id = factor_id(RP_LOCALHOST, &credential);
    if let Ceremony::Unlock { factors } = ceremony {
        if !factors.contains(&id) {
            bail!("the browser used a passkey that is not enrolled ({id})");
        }
    }
    Ok(PasskeySecret { id, secret })
}

/// Run `ceremony` in the browser on a loopback page and wait up to `timeout` for it.
pub async fn run_on_loopback(ceremony: Ceremony, timeout: Duration) -> Result<PasskeySecret> {
    let mut raw = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut raw);
    let nonce = hex::encode(raw);
    rand::rngs::OsRng.fill_bytes(&mut raw);
    let script_nonce = hex::encode(raw);
    let (sender, receiver) = oneshot::channel();
    let state = std::sync::Arc::new(Waiting {
        page: page(
            &ceremony,
            RP_LOCALHOST,
            &format!("/{nonce}/result"),
            &script_nonce,
        ),
        nonce: nonce.clone(),
        script_nonce,
        sender: Mutex::new(Some(sender)),
    });
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .context("bind a loopback port for the passkey page")?;
    let port = listener.local_addr()?.port();
    let app = Router::new()
        .route("/{nonce}", get(serve_page))
        .route("/{nonce}/result", post(take_result))
        .with_state(state);
    let (stop, stopped) = oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });

    // localhost, not 127.0.0.1: the relying party id is a host name.
    let url = format!("http://{RP_LOCALHOST}:{port}/{nonce}");
    eprintln!("Opening {url} — finish the passkey step there.");
    open_browser(&url);
    let outcome = tokio::time::timeout(timeout, receiver).await;
    let _ = stop.send(());
    let _ = server.await;
    let result = outcome
        .map_err(|_| anyhow!("no passkey result within {}s", timeout.as_secs()))?
        .map_err(|_| anyhow!("the passkey page closed without a result"))?;
    accept(&ceremony, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_factor_id_round_trips_and_rejects_foreign_shapes() {
        let id = factor_id("localhost", "AbC-_9");
        assert_eq!(parse_factor_id(&id), Some(("localhost", "AbC-_9")));
        assert_eq!(parse_factor_id("se-0011"), None);
        assert_eq!(parse_factor_id("passkey::x"), None);
    }

    #[test]
    fn unlock_accepts_only_an_enrolled_credential_and_a_full_secret() {
        let enrolled = factor_id(RP_LOCALHOST, "cred-a");
        let ceremony = Ceremony::Unlock {
            factors: vec![enrolled.clone()],
        };
        let post = |credential: &str, prf: Vec<u8>| PageResult {
            credential_id: Some(credential.to_string()),
            prf: Some(b64u(&prf)),
            error: None,
        };
        assert_eq!(
            accept(&ceremony, post("cred-a", vec![7; 32])).unwrap().id,
            enrolled
        );
        assert!(accept(&ceremony, post("cred-b", vec![7; 32])).is_err());
        assert!(accept(&ceremony, post("cred-a", vec![7; 16])).is_err());
        let failed = PageResult {
            credential_id: None,
            prf: None,
            error: Some("NotAllowedError".into()),
        };
        assert!(accept(&ceremony, failed).is_err());
    }

    #[test]
    fn the_page_offers_only_this_relying_partys_credentials_and_cannot_be_broken_out_of() {
        let ceremony = Ceremony::Unlock {
            factors: vec![
                factor_id("localhost", "here"),
                factor_id("veld.example", "there"),
            ],
        };
        let html = page(&ceremony, "localhost", "/n/result", "abc");
        assert!(html.contains(r#""allow":["here"]"#));
        assert!(!html.contains("there"));
        let hostile = Ceremony::Enrol {
            label: "</script><script>alert(1)</script>".into(),
        };
        let html = page(&hostile, "localhost", "/n/result", "abc");
        assert!(!html.contains("</script><script>alert"));
    }
}
