//! Key and login management over HTTP, through the real router: an admin with 2FA
//! steps up once per window, opens the keyring with a factor in the body, and every
//! change lands in the audit trail.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
    Router,
};
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;
use veld::{
    config::ServerConfig,
    handlers::{build_protected_routes, MultiUserMemoryManager},
};

const PASS: &str = "security-api-correct-horse-S1";

struct Api {
    app: Router,
}

struct Reply {
    status: StatusCode,
    body: Value,
    text: String,
    csp: Option<String>,
}

impl Api {
    async fn call(
        &self,
        method: Method,
        uri: &str,
        token: Option<&str>,
        step_up: Option<&str>,
        body: Option<Value>,
    ) -> Reply {
        let mut req = Request::builder().method(method).uri(uri);
        if let Some(t) = token {
            req = req.header("authorization", format!("Bearer {t}"));
        }
        if let Some(c) = step_up {
            req = req.header("x-veld-step-up", c);
        }
        let req = match body {
            Some(b) => req
                .header("content-type", "application/json")
                .body(Body::from(b.to_string())),
            None => req.body(Body::empty()),
        }
        .unwrap();
        let resp = self.app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let csp = resp
            .headers()
            .get("content-security-policy")
            .map(|v| v.to_str().unwrap().to_string());
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8_lossy(&bytes).into_owned();
        Reply {
            status,
            body: serde_json::from_slice(&bytes).unwrap_or(Value::Null),
            text,
            csp,
        }
    }

    async fn get(&self, uri: &str, token: &str) -> Reply {
        self.call(Method::GET, uri, Some(token), None, None).await
    }

    async fn post(&self, uri: &str, token: &str, body: Value) -> Reply {
        self.call(Method::POST, uri, Some(token), None, Some(body))
            .await
    }

    async fn post_up(&self, uri: &str, token: &str, code: &str, body: Value) -> Reply {
        self.call(Method::POST, uri, Some(token), Some(code), Some(body))
            .await
    }

    async fn login(&self, username: &str, password: &str, totp: Option<String>) -> Reply {
        self.call(
            Method::POST,
            "/api/user_auth/login",
            None,
            None,
            Some(json!({ "username": username, "password": password, "totp": totp })),
        )
        .await
    }
}

fn ok(reply: &Reply, what: &str) {
    assert!(
        reply.status.is_success(),
        "{what}: {} {}",
        reply.status,
        reply.text
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_admin_with_2fa_manages_keys_and_logins_over_http() {
    let dir = TempDir::new().unwrap();
    let storage = dir.path().join("data");
    std::fs::create_dir_all(&storage).unwrap();
    std::env::set_var("VELD_USER_AUTH_ENABLED", "true");
    std::env::set_var("VELD_API_KEYS", "security-admin-api-test-key");
    std::env::set_var("VELD_LAZY_CROSS_ENCODER", "1");
    std::env::set_var("VELD_MASTER_PASSPHRASE", PASS);
    std::env::set_var(
        "VELD_KEYSTORE_PATH",
        dir.path().join("global-keystore.json"),
    );
    std::env::remove_var("VELD_KEYRING_PATH");
    std::env::remove_var("VELD_KMS_WRAP_KEY");

    veld::keystore::open_record_keys(&storage.join("alice").join("storage").join("keystore.json"))
        .unwrap()
        .expect("a tenant keystore before the keyring");
    let cfg = ServerConfig {
        storage_path: storage.clone(),
        backup_enabled: false,
        ..ServerConfig::default()
    };
    let mgr = Arc::new(MultiUserMemoryManager::new(storage.clone(), cfg).unwrap());
    let api = Api {
        app: build_protected_routes(mgr, false)
            .layer(axum::middleware::from_fn(veld::auth::auth_middleware)),
    };

    // An admin with 2FA.
    ok(
        &api.call(
            Method::POST,
            "/api/user_auth/register",
            None,
            None,
            Some(json!({ "username": "root", "password": "root-password-1" })),
        )
        .await,
        "register the first admin",
    );
    let token = api.login("root", "root-password-1", None).await.body["session_token"]
        .as_str()
        .unwrap()
        .to_string();
    let enroll = api
        .post("/api/user_auth/2fa/enroll", &token, json!({}))
        .await;
    ok(&enroll, "enrol 2FA");
    let totp = totp_rs::TOTP::from_url(enroll.body["provisioning_uri"].as_str().unwrap()).unwrap();
    ok(
        &api.post(
            "/api/user_auth/2fa/confirm",
            &token,
            json!({ "totp": totp.generate_current().unwrap() }),
        )
        .await,
        "confirm 2FA",
    );

    // Status needs no step-up; a change does.
    let status = api.get("/api/user_auth/admin/keys/status", &token).await;
    ok(&status, "key status");
    assert!(status.body["status"]["keyring"].is_null());
    let scopes: Vec<_> = status.body["status"]["keystores"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k["scope"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(scopes, ["global", "alice"]);
    let refused = api
        .post(
            "/api/user_auth/admin/keys/init",
            &token,
            json!({ "passphrase": PASS }),
        )
        .await;
    assert_eq!(refused.body["code"], "STEP_UP_REQUIRED", "{}", refused.text);

    // A code steps up one session once; a second session cannot reuse it. Unlocking
    // an account is a cheap change, so the code is still inside its window.
    let second = api
        .login(
            "root",
            "root-password-1",
            Some(totp.generate_current().unwrap()),
        )
        .await;
    let token2 = second.body["session_token"].as_str().unwrap().to_string();
    let code = totp.generate_current().unwrap();
    ok(
        &api.post_up(
            "/api/user_auth/admin/users/unlock",
            &token,
            &code,
            json!({ "username": "root" }),
        )
        .await,
        "step up",
    );
    let replay = api
        .post_up(
            "/api/user_auth/admin/users/unlock",
            &token2,
            &code,
            json!({ "username": "root" }),
        )
        .await;
    assert_eq!(replay.body["code"], "STEP_UP_REPLAYED", "{}", replay.text);

    let init = api
        .post(
            "/api/user_auth/admin/keys/init",
            &token,
            json!({ "passphrase": PASS }),
        )
        .await;
    ok(&init, "keyring init in the elevated session");
    assert_eq!(init.body["enrolled"].as_array().unwrap().len(), 2);
    let first_code = init.body["recovery"]["code"].as_str().unwrap().to_string();

    // The stepped-up session works on without another code for a while.
    let issued = api
        .post(
            "/api/user_auth/admin/keys/recovery_codes",
            &token,
            json!({ "unlock": { "passphrase": PASS }, "label": "drawer" }),
        )
        .await;
    ok(&issued, "second recovery code");
    assert!(issued.body["kit"]
        .as_str()
        .unwrap()
        .contains("VELD RECOVERY KIT"));
    let second_id = issued.body["id"].as_str().unwrap().to_string();
    ok(
        &api.post(
            "/api/user_auth/admin/keys/revoke",
            &token,
            json!({ "unlock": { "recovery_code": first_code }, "id": second_id }),
        )
        .await,
        "revoke with the first code",
    );
    let wrong = api
        .post(
            "/api/user_auth/admin/keys/recovery_codes",
            &token,
            json!({ "unlock": { "passphrase": "not the passphrase" } }),
        )
        .await;
    assert_eq!(wrong.status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(wrong.body["code"], "KEY_OPERATION_REFUSED");

    // A passkey enrolled through the server's page, then used to unlock.
    let start = api
        .post(
            "/api/user_auth/admin/keys/passkey",
            &token,
            json!({ "unlock": { "passphrase": PASS }, "label": "YubiKey" }),
        )
        .await;
    ok(&start, "start a passkey enrolment");
    let path = start.body["path"].as_str().unwrap().to_string();
    let ticket = start.body["ticket"].as_str().unwrap().to_string();
    let page = api.call(Method::GET, &path, None, None, None).await;
    assert_eq!(page.status, StatusCode::OK);
    assert!(
        page.text.contains("Add a passkey to the Veld keyring")
            || page.text.contains("\"mode\":\"enrol\"")
    );
    assert!(page
        .csp
        .as_deref()
        .unwrap_or_default()
        .contains("script-src 'nonce-"));
    let prf = [9u8; 32];
    let posted = api
        .call(
            Method::POST,
            &format!("{path}/result"),
            None,
            None,
            Some(json!({
                "credential_id": "Y3JlZA",
                "prf": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(prf),
            })),
        )
        .await;
    assert_eq!(posted.status, StatusCode::NO_CONTENT, "{}", posted.text);
    let poll = api
        .get(
            &format!("/api/user_auth/admin/keys/passkey/{ticket}"),
            &token,
        )
        .await;
    assert_eq!(poll.body["state"], "done", "{}", poll.text);
    let passkey_id = poll.body["factor"]["id"].as_str().unwrap().to_string();
    assert_eq!(passkey_id, "passkey:localhost:Y3JlZA");
    let again = api.call(Method::GET, &path, None, None, None).await;
    assert_eq!(
        again.status,
        StatusCode::NOT_FOUND,
        "a ticket is single-use"
    );
    let rotated = api
        .post(
            "/api/user_auth/admin/keys/rotate_data_keys",
            &token,
            json!({ "unlock": { "passkey": {
                "id": passkey_id,
                "secret": base64::engine::general_purpose::STANDARD.encode(prf),
            } } }),
        )
        .await;
    ok(&rotated, "rotate data keys with the passkey");
    assert!(rotated
        .body
        .as_array()
        .unwrap()
        .iter()
        .all(|e| e["epoch"] == 1));

    // Logins: a user is disabled, refused, enabled, and given a temporary password.
    ok(
        &api.post(
            "/api/user_auth/register",
            &token,
            json!({ "username": "bob", "password": "bob-password-1" }),
        )
        .await,
        "add bob",
    );
    let bob = api.login("bob", "bob-password-1", None).await.body["session_token"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        api.get("/api/user_auth/admin/users", &bob).await.status,
        StatusCode::FORBIDDEN,
        "not an admin"
    );
    let users = api.get("/api/user_auth/admin/users", &token).await;
    assert_eq!(users.body.as_array().unwrap().len(), 2);
    ok(
        &api.post(
            "/api/user_auth/admin/users/disable",
            &token,
            json!({ "username": "bob" }),
        )
        .await,
        "disable bob",
    );
    assert_eq!(
        api.get("/api/user_auth/me", &bob).await.status,
        StatusCode::UNAUTHORIZED,
        "his sessions ended"
    );
    let refused = api.login("bob", "bob-password-1", None).await;
    assert_eq!(refused.body["code"], "ACCOUNT_DISABLED", "{}", refused.text);
    ok(
        &api.post(
            "/api/user_auth/admin/users/enable",
            &token,
            json!({ "username": "bob" }),
        )
        .await,
        "enable bob",
    );
    let reset = api
        .post(
            "/api/user_auth/admin/users/reset_password",
            &token,
            json!({ "username": "bob" }),
        )
        .await;
    ok(&reset, "reset bob's password");
    let temporary = reset.body["temporary_password"].as_str().unwrap();
    ok(
        &api.login("bob", temporary, None).await,
        "sign in with the temporary password",
    );
    assert_eq!(
        api.login("bob", "bob-password-1", None).await.status,
        StatusCode::UNAUTHORIZED
    );

    let audit = api.get("/api/user_auth/admin/audit?limit=50", &token).await;
    ok(&audit, "audit");
    let actions: Vec<_> = audit
        .body
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["action"].as_str().unwrap().to_string())
        .collect();
    for expected in [
        "keyring_init",
        "recovery_code_issued",
        "key_factor_revoked",
        "key_factor_enrolled",
        "data_keys_rotated",
        "user_disable",
        "user_enable",
        "password_reset_by_admin",
    ] {
        assert!(
            actions.iter().any(|a| a == expected),
            "{expected} missing from {actions:?}"
        );
    }
    assert_eq!(actions[0], "password_reset_by_admin", "newest first");
}
