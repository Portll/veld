//! `POST /api/admin/storage/{export,import,migrate}` — the CLI's migration commands over HTTP.
//!
//! Gated by the same admin key as `/api/admin/reset-rate-limit`, checked before the body
//! is parsed. Dumps live under `<data>/exports/<name>`: a request names a directory, never
//! a path, so the admin key does not become a write-anywhere primitive. Writing into the
//! backend this server is serving is refused — a request that lazily opens a user's store
//! mid-import would race it — and is left to the offline CLI.

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

use super::admin::require_admin_key;
use super::router::AppState;
use crate::config::StorageBackend;
use crate::storage::migrate::{self, Scope};

const EXPORTS_DIR: &str = "exports";
const QUARANTINE_DIR: &str = "quarantine";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportRequest {
    pub user: String,
    pub name: String,
    /// Defaults to the backend this server is serving.
    pub backend: Option<String>,
    /// Opt-in: set undecodable records aside under `<data>/quarantine/<name>`.
    pub quarantine: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportRequest {
    pub name: String,
    pub backend: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MigrateRequest {
    pub from: String,
    pub to: String,
    #[serde(default = "all_users")]
    pub user: String,
    #[serde(default)]
    pub dry_run: bool,
    /// Opt-in: set undecodable records aside under `<data>/quarantine/<name>`.
    pub quarantine: Option<String>,
}

fn all_users() -> String {
    "all".to_string()
}

fn respond(status: StatusCode, body: serde_json::Value) -> Response {
    (status, Json(body)).into_response()
}

fn bad_request(detail: impl std::fmt::Display) -> Response {
    respond(
        StatusCode::BAD_REQUEST,
        json!({"error": "bad_request", "detail": detail.to_string()}),
    )
}

fn failed(e: anyhow::Error) -> Response {
    respond(
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({"error": "storage_operation_failed", "detail": format!("{e:#}")}),
    )
}

fn parse<T: serde::de::DeserializeOwned>(body: &Bytes) -> Result<T, Response> {
    serde_json::from_slice(body).map_err(bad_request)
}

fn backend(value: &str) -> Result<StorageBackend, Response> {
    value.parse::<StorageBackend>().map_err(bad_request)
}

fn export_dir(state: &AppState, name: &str) -> Result<PathBuf, Response> {
    crate::path_guard::sanitize_component(name, "export name").map_err(bad_request)?;
    Ok(state.base_path.join(EXPORTS_DIR).join(name))
}

fn quarantine_dir(state: &AppState, name: Option<&str>) -> Result<Option<PathBuf>, Response> {
    name.map(|n| {
        crate::path_guard::sanitize_component(n, "quarantine name").map_err(bad_request)?;
        Ok(state.base_path.join(QUARANTINE_DIR).join(n))
    })
    .transpose()
}

fn refuse_live_target(state: &AppState, target: StorageBackend) -> Result<(), Response> {
    if target == state.server_config.effective_storage_backend {
        return Err(respond(
            StatusCode::CONFLICT,
            json!({
                "error": "target_is_serving",
                "detail": format!(
                    "this server is serving {target}; writing into it would race live requests. \
                     Stop the server and run `veld import` / `veld migrate` offline."
                ),
            }),
        ));
    }
    Ok(())
}

async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> anyhow::Result<T> + Send + 'static,
) -> anyhow::Result<T> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| anyhow::anyhow!("storage task did not complete: {e}"))?
}

pub async fn export(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(refusal) = require_admin_key(&headers) {
        return refusal;
    }
    let prepared = (|| {
        let req: ExportRequest = parse(&body)?;
        let scope = Scope::parse(&req.user).map_err(bad_request)?;
        let source = match req.backend.as_deref() {
            Some(b) => backend(b)?,
            None => state.server_config.effective_storage_backend,
        };
        let quarantine = quarantine_dir(&state, req.quarantine.as_deref())?;
        Ok::<_, Response>((scope, source, export_dir(&state, &req.name)?, quarantine))
    })();
    let (scope, source, out, quarantine) = match prepared {
        Ok(p) => p,
        Err(r) => return r,
    };
    let base = state.base_path.clone();
    match blocking(move || migrate::export(&base, source, &scope, &out, quarantine.as_deref()))
        .await
    {
        Ok(reports) => respond(StatusCode::OK, json!({"dumps": reports})),
        Err(e) => failed(e),
    }
}

pub async fn import(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(refusal) = require_admin_key(&headers) {
        return refusal;
    }
    let prepared = (|| {
        let req: ImportRequest = parse(&body)?;
        let target = backend(&req.backend)?;
        refuse_live_target(&state, target)?;
        Ok::<_, Response>((target, export_dir(&state, &req.name)?))
    })();
    let (target, input) = match prepared {
        Ok(p) => p,
        Err(r) => return r,
    };
    let base = state.base_path.clone();
    match blocking(move || migrate::import(&base, target, &input)).await {
        Ok(reports) => respond(StatusCode::OK, json!({"imported": reports})),
        Err(e) => failed(e),
    }
}

pub async fn migrate(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(refusal) = require_admin_key(&headers) {
        return refusal;
    }
    let prepared = (|| {
        let req: MigrateRequest = parse(&body)?;
        let from = backend(&req.from)?;
        let to = backend(&req.to)?;
        if !req.dry_run {
            refuse_live_target(&state, to)?;
        }
        let scope = Scope::parse(&req.user).map_err(bad_request)?;
        let quarantine = quarantine_dir(&state, req.quarantine.as_deref())?;
        Ok::<_, Response>((from, to, scope, req.dry_run, quarantine))
    })();
    let (from, to, scope, dry_run, quarantine) = match prepared {
        Ok(p) => p,
        Err(r) => return r,
    };
    let base = state.base_path.clone();
    match blocking(move || {
        migrate::migrate(&base, from, to, &scope, dry_run, quarantine.as_deref())
    })
    .await
    {
        Ok(report) if report.blocked() => respond(
            StatusCode::CONFLICT,
            json!({"error": "blocked", "report": report}),
        ),
        Ok(report) => respond(StatusCode::OK, json!({"report": report})),
        Err(e) => failed(e),
    }
}

#[cfg(test)]
mod tests;
